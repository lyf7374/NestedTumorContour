import torch
from torch import nn
import torch.nn.functional as F
import math


from mmcv.runner.fp16_utils import force_fp32

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
from mmseg.models.losses import accuracy
from utils.GroupViT import GroupViTEncoder

def trunc_normal_init(module: nn.Module, mean: float = 0, std: float = 1, a: float = -2, b: float = 2, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TPN_Decoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = None

    def forward(self, tgt, memory):
        output = tgt
        for mod in self.layers:
            output, attn = mod(output, memory)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn

class TPN_DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):
        super(TPN_DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.multihead_attn = Attention(d_model, num_heads=nhead, qkv_bias=True, attn_drop=0.1)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.activation = F.relu

    def forward(self, tgt, memory):
        # Ensure tgt and memory have shape [batch_size, seq_length, embed_dim]
        # Remove unnecessary transpositions
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Pass tgt and memory directly without transposing
        tgt2, attn = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size(1)

        q = self.q(xq).reshape(B, Nq, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k(xk).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v(xv).reshape(B, Nk, self.num_heads, C // self.num_heads).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_save.mean(dim=1)

# @HEADS.register_module()
class ATMHead(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            num_classes,
            embed_dims=384,
            num_layers=3,
            num_heads=8,
            use_proj=True,
            CE_loss=False,
            **kwargs,
    ):
        super(ATMHead, self).__init__(in_channels=in_channels, channels=embed_dims, num_classes=num_classes, **kwargs)

        self.image_size = img_size
        dim = embed_dims

        # Input projection layer
        if use_proj:
            self.input_proj = nn.Linear(self.in_channels, dim)
            trunc_normal_(self.input_proj.weight, std=.02)
            self.proj_norm = nn.LayerNorm(dim)
        else:
            self.input_proj = nn.Identity()
            self.proj_norm = nn.Identity()

        # Decoder
        decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=num_heads, dim_feedforward=dim * 4)
        self.decoder = TPN_Decoder(decoder_layer, num_layers)

        # Query embeddings
        self.q = nn.Embedding(num_classes, dim)

        # Classifier
        self.class_embed = nn.Linear(dim, num_classes + 1)  # +1 for null class
        self.CE_loss = CE_loss
        delattr(self, 'conv_seg')

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)
    def compute_hw(self, N):
        for h in range(int(N ** 0.5), 0, -1):
            if N % h == 0:
                w = N // h
                return h, w
        return N, 1  # If N is prime
    def forward(self, feature_map, group_tokens):
        # feature_map: [batch_size, num_tokens, embed_dim]
        # group_tokens: [batch_size, num_group_tokens, embed_dim]

        bs = feature_map.size(0)
        num_tokens = feature_map.size(1)

        # Process feature_map
        x = self.input_proj(feature_map)  # [batch_size, num_tokens, dim]
        x = self.proj_norm(x)             # [batch_size, num_tokens, dim]

        # Prepare query embeddings
        q = self.q.weight.unsqueeze(0).repeat(bs, 1, 1)  # [batch_size, num_queries, dim]

        # Combine x and group_tokens
        combined_memory = torch.cat([x, group_tokens], dim=1)  # [batch_size, num_tokens + num_group_tokens, dim]

        # Pass through decoder
        q, attn = self.decoder(q, combined_memory)  # Ensure decoder expects batch-first tensors

        # Rest of the code remains the same
        # Classify
        outputs_class = self.class_embed(q)  # [batch_size, num_queries, num_classes + 1]

        # Process attention maps
        attn_image_tokens = attn[:, :, :num_tokens]  # [batch_size, num_queries, num_tokens]
        # Use only attn_image_tokens
        attn_combined = attn_image_tokens  # [batch_size, num_queries, num_tokens]

        # Reshape attn_combined to spatial dimensions
        N = num_tokens
        h, w = self.compute_hw(N)
        attn_combined = attn_combined.view(bs, self.num_classes, h, w)  # Correct shape

        # Upsample to image size
        attn_combined = F.interpolate(attn_combined, size=(self.image_size, self.image_size),
                                      mode='bilinear', align_corners=False)

        # Compute final segmentation map
        mask_cls = F.softmax(outputs_class, dim=-1)[..., :-1]  # [batch_size, num_queries, num_classes]
        mask_pred = attn_combined.sigmoid()  # [batch_size, num_queries, h, w]
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)  # [batch_size, num_classes, h, w]

        # Return output
        if self.training:
            out = {"pred_logits": outputs_class, "pred_masks": mask_pred, "pred": semseg}
            return out
        else:
            return semseg

    # @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        if self.CE_loss:
            return super().losses(seg_logit["pred"], seg_label)

        if isinstance(seg_logit, dict):
            seg_label = seg_label.squeeze(1)
            loss = self.loss_decode(seg_logit, seg_label, ignore_index=self.ignore_index)
            loss['acc_seg'] = accuracy(seg_logit["pred"], seg_label, ignore_index=self.ignore_index)
            return loss
class GroupViTSegmenterATM(nn.Module):
    def __init__(self, num_classes, img_size=224, in_chans=4,  patch_size=4, **kwargs):
        super(GroupViTSegmenterATM, self).__init__()
        self.encoder = GroupViTEncoder(img_size=img_size, in_chans=in_chans, patch_size=patch_size, **kwargs)
        self.decoder = ATMHead(
            img_size=img_size,
            in_channels=self.encoder.width,
            num_classes=num_classes,
            embed_dims=self.encoder.width,
            CE_loss=False
        )

    def forward(self, x):
        # Get feature maps and group tokens from encoder
        feature_map, group_tokens, _ = self.encoder(x, return_group_token=True, return_attn=True)
        # feature_map: [batch_size, num_tokens, embed_dim]
        # group_tokens: [batch_size, num_group_tokens, embed_dim]

        # Pass the encoder outputs to the decoder
        seg_output = self.decoder(feature_map, group_tokens)
        return seg_output


