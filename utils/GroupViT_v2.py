from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# from mmcv.utils import Registry
# from omegaconf import OmegaConf

# MODELS = Registry('model')


# def build_model(config):

#     model = MODELS.build(OmegaConf.to_container(config, resolve=True))

#     return model


import math

import torch.nn.functional as F


class Result:

    def __init__(self, as_dict=False):
        if as_dict:
            self.outs = {}
        else:
            self.outs = []

    @property
    def as_dict(self):
        return isinstance(self.outs, dict)

    def append(self, element, name=None):
        if self.as_dict:
            assert name is not None
            self.outs[name] = element
        else:
            self.outs.append(element)

    def update(self, **kwargs):
        if self.as_dict:
            self.outs.update(**kwargs)
        else:
            for v in kwargs.values():
                self.outs.append(v)

    def as_output(self):
        if self.as_dict:
            return self.outs
        else:
            return tuple(self.outs)

    def as_return(self):
        outs = self.as_output()
        if self.as_dict:
            return outs
        if len(outs) == 1:
            return outs[0]
        return outs


def interpolate_pos_encoding(pos_embed, H, W):
    num_patches = H * W

    N = pos_embed.shape[1]
    if num_patches == N and W == H:
        return pos_embed
    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    patch_pos_embed = F.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        size=(H, W),
        mode='bicubic',
        align_corners=False)
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerMlp(Mlp):

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict

    def extra_repr(self):
        return f'num_heads: {self.num_heads}, \n' \
               f'hard: {self.hard}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'gumbel_tau: {self.gumbel_tau}, \n' \
               f'assign_eps: {self.assign_eps}'


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 *,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 num_output_group,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.):
        super(GroupingBlock, self).__init__()
        self.dim = dim
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.num_output_group = num_output_group
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_inter = Mlp(num_group_token, tokens_dim, num_output_group)
        self.norm_post_tokens = norm_layer(dim)
        # norm on x
        self.norm_x = norm_layer(dim)
        self.pre_assign_attn = CrossAttnBlock(
            dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self):
        return f'hard={self.hard}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'sum_assign={self.sum_assign}, \n' \
               f'num_output_group={self.num_output_group}, \n '

    def project_group_token(self, group_tokens):
        """
        Args:
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of
            group tokens, it's already softmaxed along dim=-1

        Returns:
            projected_group_tokens (torch.Tensor): [B, S_2, C]
        """
        # [B, S_2, C] <- [B, S_1, C]
        projected_group_tokens = self.mlp_inter(group_tokens.transpose(1, 2)).transpose(1, 2)
        projected_group_tokens = self.norm_post_tokens(projected_group_tokens)
        return projected_group_tokens

    def forward(self, x, group_tokens, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of
                group tokens
        """
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        projected_group_tokens = self.project_group_token(group_tokens)
        projected_group_tokens = self.pre_assign_attn(projected_group_tokens, x)
        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, attn_dict


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self):
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class AttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            qkv_fuse=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GroupingLayer(nn.Module):
    """A Transformer layer with Grouping Block for one stage.

    Args:
        dim (int): Number of input channels.
        num_input_token (int): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer.
            In GroupViT setting, Grouping Block serves as the downsampling layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        group_projector (nn.Module | None, optional): Projector for the grouping layer. Default: None.
        zero_init_group_token (bool): Whether to initialize the grouping token to 0. Default: False.
    """

    def __init__(self,
                 dim,
                 num_input_token,
                 depth,
                 num_heads,
                 num_group_token,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 group_projector=None,
                 zero_init_group_token=False):

        super().__init__()
        self.dim = dim
        self.input_length = num_input_token
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None

        # build blocks
        self.depth = depth
        blocks = []
        for i in range(depth):
            blocks.append(
                AttnBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i],
                    norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)

        self.downsample = downsample
        self.input_resolution = num_input_token
        self.use_checkpoint = use_checkpoint

        self.group_projector = group_projector

    @property
    def with_group_token(self):
        return self.group_token is not None

    def extra_repr(self):
        return f'dim={self.dim}, \n' \
               f'input_resolution={self.input_resolution}, \n' \
               f'depth={self.depth}, \n' \
               f'num_group_token={self.num_group_token}, \n'

    def split_x(self, x):
        if self.with_group_token:
            return x[:, :-self.num_group_token], x[:, -self.num_group_token:]
        else:
            return x, None

    def concat_x(self, x, group_token=None):
        if group_token is None:
            return x
        return torch.cat([x, group_token], dim=1)

    def forward(self, x, prev_group_token=None, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_group_token (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention maps
        """
        if self.with_group_token:
            group_token = self.group_token.expand(x.size(0), -1, -1)
            if self.group_projector is not None:
                group_token = group_token + self.group_projector(prev_group_token)
        else:
            group_token = None

        B, L, C = x.shape
        cat_x = self.concat_x(x, group_token)
        for blk_idx, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                cat_x = checkpoint.checkpoint(blk, cat_x)
            else:
                cat_x = blk(cat_x)

        x, group_token = self.split_x(cat_x)

        attn_dict = None
        if self.downsample is not None:
            x, attn_dict = self.downsample(x, group_token, return_attn=return_attn)

        return x, group_token, attn_dict


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
            int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[1] + 1),
            int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[0] + 1),
        )

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape

class GroupViTEncoder(nn.Module):
    r""" Group Vision Transformer Encoder
        A modified version of GroupViT without average pooling and classification head,
        suitable for contrastive learning.
    
    Args:
        img_size (int | tuple[int]): Input image size. Default 224
        patch_size (int | tuple[int]): Patch size. Default: 16
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 384
        embed_factors (list[int]): Embedding dim multipliers for each stage.
        depths (list[int]): Depth of each stage
        num_heads (list[int]): Number of heads for each stage
        num_group_tokens (list[int]): Number of group tokens for each stage
        num_output_group (list[int]): Number of output groups for each stage
        hard_assignment (bool): Whether to use hard assignment or not. Default: True
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pos_embed_type (str): Type of positional embedding. Default: 'simple'
        freeze_patch_embed (bool): Whether to freeze patch embedding. Default: False
    """
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384, 
                 embed_factors=[1, 1, 1, 1], depths=[3, 3, 3, 3], num_heads=[6, 6, 6, 6],
                 num_group_tokens=[64, 32, 16, 8], num_output_groups=[64, 32, 16], 
                 hard_assignment=True, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 patch_norm=True, use_checkpoint=False, pos_embed_type='simple',
                 freeze_patch_embed=False):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert len(embed_factors) == len(depths) == len(num_group_tokens)
        assert all(_ == 0 for _ in num_heads) or len(depths) == len(num_heads)
        assert len(depths) - 1 == len(num_output_groups)

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.pos_embed_type = pos_embed_type
        assert pos_embed_type in ['simple', 'fourier']

        norm_layer = nn.LayerNorm

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Remove average pooling
        # self.avgpool = nn.AdaptiveAvgPool1d(1)  # Removed

        if pos_embed_type == 'simple':
            self.pos_embed = self.build_simple_position_embedding()
        elif pos_embed_type == 'fourier':
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            raise ValueError

        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False

        self.pos_drop = nn.Dropout(p=drop_rate)
        # Update the number of layers
        self.num_layers = len(depths)
        # Update num_features to match the last embedding dimension
        self.num_features = int(embed_dim * embed_factors[-1])
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Collect the feature dimensions at each stage
        self.feature_dims = []
        # Build layers
        num_input_token = self.patch_embed.num_patches
        num_output_token = num_input_token
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * embed_factors[i_layer])
            self.feature_dims.append(dim)  # Store feature dimension

            downsample = None
            if i_layer < self.num_layers - 1:
                out_dim = int(embed_dim * embed_factors[i_layer + 1])
                downsample = GroupingBlock(
                    dim=dim,
                    out_dim=out_dim,
                    num_heads=num_heads[i_layer],
                    num_group_token=num_group_tokens[i_layer],
                    num_output_group=num_output_groups[i_layer],
                    norm_layer=norm_layer,
                    hard=hard_assignment,
                    gumbel=hard_assignment)
                num_output_token = num_output_groups[i_layer]

            # Adjust group_projector if necessary
            if i_layer > 0 and num_group_tokens[i_layer] > 0:
                prev_dim = int(embed_dim * embed_factors[i_layer - 1])
                group_projector = nn.Sequential(
                    norm_layer(prev_dim),
                    MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))
                if dim != prev_dim:
                    group_projector = nn.Sequential(
                        group_projector,
                        norm_layer(prev_dim),
                        nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None

            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                zero_init_group_token=group_projector is not None)
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                num_input_token = num_output_token
        # Remove normalization and head
        self.norm = norm_layer(self.num_features)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # Removed

        self.apply(self._init_weights)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        if self.pos_embed_type == 'simple' and 'pos_embed' in state_dict:
            load_pos_embed = state_dict['pos_embed']
            pos_embed = self.pos_embed
            if load_pos_embed.shape != pos_embed.shape:
                H_new = int(self.patch_embed.num_patches**0.5)
                W_new = H_new
                H_ori = int(load_pos_embed.shape[1]**0.5)
                W_ori = H_ori
                load_pos_embed = F.interpolate(
                    rearrange(load_pos_embed, 'b (h w) c -> b c h w', h=H_ori, w=W_ori, b=1),
                    size=(H_new, W_new),
                    mode='bicubic',
                    align_corners=False)
                load_pos_embed = rearrange(load_pos_embed, 'b c h w -> b (h w) c', h=H_new, w=W_new)
                state_dict['pos_embed'] = load_pos_embed
        return super().load_state_dict(state_dict, strict)

    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.patches_resolution
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False
        return pos_embed

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H, W)
        return pos_embed

    def forward_features(self, x, *, return_attn=False):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        x = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(x)

        group_token = None
        attn_dict_list = []
        features = []

        # Collect initial features after patch embedding
        features.append((x, hw_shape))

        for layer in self.layers:
            x, group_token, attn_dict = layer(x, group_token, return_attn=return_attn)
            attn_dict_list.append(attn_dict)
            N = x.shape[1]
            H, W = self.compute_hw(N)
            features.append((x, (H, W)))  # Collect features at each layer

        x = self.norm(x)
        features.append((x, (H, W)))  # Collect the final feature

        return x, group_token, attn_dict_list, features

    def compute_hw(self, N):
        # Compute H and W from the number of tokens N
        for H in range(int(N ** 0.5), 0, -1):
            if N % H == 0:
                W = N // H
                return H, W
        return N, 1  # If N is prime

    def forward(self, x, *, return_attn=False, return_group_token=False):
        x, group_token, attn_dicts, features = self.forward_features(x, return_attn=return_attn)
        
        if return_attn and return_group_token:
            return x, group_token, attn_dicts, features
        elif return_attn:
            return x, attn_dicts, features
        elif return_group_token:
            return x, group_token, features
        else:
            return x, features
        

        
class UNetDecoder(nn.Module):
    def __init__(self, feature_channels, num_classes):
        super(UNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)

        # Adjusted to 5 upsampling layers
        self.up5 = nn.ConvTranspose2d(
            feature_channels[5], feature_channels[4], kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(
            feature_channels[4] + feature_channels[4], feature_channels[4], kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(feature_channels[4])

        self.up4 = nn.ConvTranspose2d(
            feature_channels[4], feature_channels[3], kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(
            feature_channels[3] + feature_channels[3], feature_channels[3], kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_channels[3])

        self.up3 = nn.ConvTranspose2d(
            feature_channels[3], feature_channels[2], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            feature_channels[2] + feature_channels[2], feature_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_channels[2])

        self.up2 = nn.ConvTranspose2d(
            feature_channels[2], feature_channels[1], kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            feature_channels[1] + feature_channels[1], feature_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_channels[1])

        self.up1 = nn.ConvTranspose2d(
            feature_channels[1], feature_channels[0], kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            feature_channels[0] + feature_channels[0], feature_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channels[0])

        self.final_conv = nn.Conv2d(feature_channels[0], num_classes, kernel_size=1)
    def forward(self, features):
        # Unpack features and their spatial dimensions
        (f0, hw0), (f1, hw1), (f2, hw2), (f3, hw3), (f4, hw4), (f5, hw5) = features

        # Start from the deepest feature
        x = f5.permute(0, 2, 1).contiguous().view(-1, f5.shape[2], *hw5)

        # Level 5
        x = self.up5(x)  # x: [B, C, H*2, W*2]
        f4 = f4.permute(0, 2, 1).contiguous().view(-1, f4.shape[2], *hw4)
        # Upsample f4 to match x's spatial dimensions
        if f4.shape[2:] != x.shape[2:]:
            f4 = F.interpolate(f4, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f4], dim=1)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # Level 4
        x = self.up4(x)
        f3 = f3.permute(0, 2, 1).contiguous().view(-1, f3.shape[2], *hw3)
        if f3.shape[2:] != x.shape[2:]:
            f3 = F.interpolate(f3, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f3], dim=1)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Level 3
        x = self.up3(x)
        f2 = f2.permute(0, 2, 1).contiguous().view(-1, f2.shape[2], *hw2)
        if f2.shape[2:] != x.shape[2:]:
            f2 = F.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f2], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Level 2
        x = self.up2(x)
        f1 = f1.permute(0, 2, 1).contiguous().view(-1, f1.shape[2], *hw1)
        if f1.shape[2:] != x.shape[2:]:
            f1 = F.interpolate(f1, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f1], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Level 1
        x = self.up1(x)
        f0 = f0.permute(0, 2, 1).contiguous().view(-1, f0.shape[2], *hw0)
        if f0.shape[2:] != x.shape[2:]:
            f0 = F.interpolate(f0, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, f0], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Final upsampling to match the input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        return x


class GroupViTSegmenter(nn.Module):
    def __init__(self, num_classes, img_size=224, **kwargs):
        super(GroupViTSegmenter, self).__init__()
        self.encoder = GroupViTEncoder(img_size=img_size, **kwargs)
        # Include the initial embedding dimension in feature_channels
        self.decoder = UNetDecoder(
            feature_channels=[self.encoder.embed_dim] + self.encoder.feature_dims + [self.encoder.num_features],
            num_classes=num_classes
        )
    def forward(self, x):
        # Obtain features from the encoder
        x, group_token, attn_dicts, features = self.encoder(x, return_group_token=True, return_attn=True)
        # Pass features to the decoder
        out = self.decoder(features)
        return out



    # model = GroupViTSegmenter(
    #     num_classes=4,
    #     img_size=224,
    #     in_chans=4,
    #     embed_dim=384,
    #     embed_factors=[1, 1, 1, 1],
    #     depths=[3, 3, 3, 3],
    #     num_heads=[6, 6, 6, 6],
    #     num_group_tokens=[64, 32, 16, 8],
    #     num_output_groups=[64, 32, 16],
    #     # ... [Include any other necessary arguments]
    # )
'''

Version_2


'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Assume necessary imports for PatchEmbed, GroupingBlock, GroupingLayer, MixerMlp, interpolate_pos_encoding, and trunc_normal_

class GroupViTEncoder(nn.Module):
    r"""Group Vision Transformer Encoder with modifications to include downsampling in the last layer.

    Args:
        img_size (int | tuple[int]): Input image size. Default 224
        patch_size (int | tuple[int]): Patch size. Default: 16
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 384
        embed_factors (list[int]): Embedding dim multipliers for each stage.
        depths (list[int]): Depth of each stage
        num_heads (list[int]): Number of heads for each stage
        num_group_tokens (list[int]): Number of group tokens for each stage
        num_output_groups (list[int]): Number of output groups for each stage
        hard_assignment (bool): Whether to use hard assignment or not. Default: True
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pos_embed_type (str): Type of positional embedding. Default: 'simple'
        freeze_patch_embed (bool): Whether to freeze patch embedding. Default: False
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384,
                 embed_factors=[1, 1, 1, 1, 1], depths=[3, 3, 3, 3, 3], num_heads=[6, 6, 6, 6, 6],
                 num_group_tokens=[64, 32, 16, 8, 4], num_output_groups=[64, 32, 16, 8, 4],
                 hard_assignment=True, mlp_ratio=4., qkv_bias=True,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 patch_norm=True, use_checkpoint=False, pos_embed_type='simple',
                 freeze_patch_embed=False):
        super().__init__()
        assert patch_size in [4, 8, 16]
        assert len(embed_factors) == len(depths) == len(num_group_tokens) == len(num_heads)
        # Adjusted assertion to include the last layer
        assert len(depths) == len(num_output_groups)

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.num_group_tokens = num_group_tokens
        self.num_output_groups = num_output_groups
        self.pos_embed_type = pos_embed_type
        assert pos_embed_type in ['simple', 'fourier']

        norm_layer = nn.LayerNorm

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Position Embedding
        if pos_embed_type == 'simple':
            self.pos_embed = self.build_simple_position_embedding()
        elif pos_embed_type == 'fourier':
            self.pos_embed = self.build_2d_sincos_position_embedding()
        else:
            raise ValueError

        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.pos_embed.requires_grad = False

        self.pos_drop = nn.Dropout(p=drop_rate)
        # Update the number of layers
        self.num_layers = len(depths)
        # Update num_features to match the last embedding dimension
        self.num_features = int(embed_dim * embed_factors[-1])
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Collect the feature dimensions at each stage
        self.feature_dims = []
        # Build layers
        num_input_token = self.patch_embed.num_patches
        num_output_token = num_input_token
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * embed_factors[i_layer])
            self.feature_dims.append(dim)  # Store feature dimension

            # Include downsampling (grouping) in all layers, including the last one
            out_dim = int(embed_dim * embed_factors[i_layer + 1]) if i_layer < self.num_layers - 1 else dim
            downsample = GroupingBlock(
                dim=dim,
                out_dim=out_dim,
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                num_output_group=num_output_groups[i_layer],
                norm_layer=norm_layer,
                hard=hard_assignment,
                gumbel=hard_assignment)
            num_output_token = num_output_groups[i_layer]

            # Adjust group_projector if necessary
            if i_layer > 0 and num_group_tokens[i_layer] > 0:
                prev_dim = int(embed_dim * embed_factors[i_layer - 1])
                group_projector = nn.Sequential(
                    norm_layer(prev_dim),
                    MixerMlp(num_group_tokens[i_layer - 1], prev_dim // 2, num_group_tokens[i_layer]))
                if dim != prev_dim:
                    group_projector = nn.Sequential(
                        group_projector,
                        norm_layer(prev_dim),
                        nn.Linear(prev_dim, dim, bias=False))
            else:
                group_projector = None

            layer = GroupingLayer(
                dim=dim,
                num_input_token=num_input_token,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                num_group_token=num_group_tokens[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                group_projector=group_projector,
                zero_init_group_token=group_projector is not None)
            self.layers.append(layer)
            num_input_token = num_output_token  # Update for next layer

        # Normalization layer
        self.norm = norm_layer(self.num_features)

        self.apply(self._init_weights)

    def load_state_dict(self, state_dict: 'OrderedDict[str, torch.Tensor]', strict: bool = True):
        if self.pos_embed_type == 'simple' and 'pos_embed' in state_dict:
            load_pos_embed = state_dict['pos_embed']
            pos_embed = self.pos_embed
            if load_pos_embed.shape != pos_embed.shape:
                H_new = int(self.patch_embed.num_patches**0.5)
                W_new = H_new
                H_ori = int(load_pos_embed.shape[1]**0.5)
                W_ori = H_ori
                load_pos_embed = F.interpolate(
                    rearrange(load_pos_embed, 'b (h w) c -> b c h w', h=H_ori, w=W_ori, b=1),
                    size=(H_new, W_new),
                    mode='bicubic',
                    align_corners=False)
                load_pos_embed = rearrange(load_pos_embed, 'b c h w -> b (h w) c', h=H_new, w=W_new)
                state_dict['pos_embed'] = load_pos_embed
        return super().load_state_dict(state_dict, strict)

    def build_simple_position_embedding(self):
        pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, self.embed_dim))
        trunc_normal_(pos_embed, std=.02)
        return pos_embed

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.patches_resolution
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        pos_embed = nn.Parameter(pos_emb)
        pos_embed.requires_grad = False
        return pos_embed

    @property
    def width(self):
        return self.num_features

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pos_embed(self, B, H, W):
        if self.training:
            return self.pos_embed
        pos_embed = self.pos_embed
        pos_embed = interpolate_pos_encoding(pos_embed, H, W)
        return pos_embed
    def forward_features(self, x, *, return_attn=False):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        x = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(x)

        group_token = None
        attn_dict_list = []
        features = []
        group_tokens_list = []

        for i, layer in enumerate(self.layers):
            x, group_token, attn_dict = layer(x, group_token, return_attn=return_attn)
            attn_dict_list.append(attn_dict)
            features.append((x, self.compute_hw(x.shape[1])))

            if group_token is not None:
                group_tokens_list.append(group_token)

        x = self.norm(x)
        # Not collecting the normalized x as a feature

        return x, group_tokens_list, attn_dict_list, features


    def compute_hw(self, N):
        # Compute H and W from the number of tokens N
        for H in range(int(N ** 0.5), 0, -1):
            if N % H == 0:
                W = N // H
                return H, W
        return N, 1  # If N is prime

    def forward(self, x, *, return_attn=False, return_group_tokens=False):
        x, group_tokens_list, attn_dicts, features = self.forward_features(x, return_attn=return_attn)

        if return_attn and return_group_tokens:
            return x, group_tokens_list, attn_dicts, features
        elif return_attn:
            return x, attn_dicts, features
        elif return_group_tokens:
            return x, group_tokens_list, features
        else:
            return x, features


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    def __init__(self, feature_dim, group_token_dim):
        super(AttentionFusion, self).__init__()
        self.feature_dim = feature_dim
        self.group_token_dim = group_token_dim

        # Linear projections
        self.query_proj = nn.Conv2d(feature_dim, feature_dim, kernel_size=1)
        self.key_proj = nn.Linear(group_token_dim, feature_dim)
        self.value_proj = nn.Linear(group_token_dim, feature_dim)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, group_tokens):
        B, C, H, W = x.size()
        N = group_tokens.size(1)

        # Project features and reshape
        query = self.query_proj(x).view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # Project group tokens
        key = self.key_proj(group_tokens)  # [B, N, C]
        value = self.value_proj(group_tokens)  # [B, N, C]

        # Compute attention
        attn = torch.bmm(query, key.transpose(1, 2))  # [B, HW, N]
        attn = self.softmax(attn / (self.feature_dim ** 0.5))

        # Apply attention
        out = torch.bmm(attn, value)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)

        # Fuse with original features
        x = x + out  # Residual connection

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDecoder(nn.Module):
    def __init__(self, feature_channels, group_token_channels, num_classes):
        super(UNetDecoder, self).__init__()
        self.num_classes = num_classes
        self.relu = nn.ReLU(inplace=True)

        # Level 5 Decoder Blocks
        self.up5 = nn.ConvTranspose2d(
            feature_channels[4], feature_channels[3], kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(
            feature_channels[3] + feature_channels[3] + group_token_channels[4],
            feature_channels[3],
            kernel_size=3,
            padding=1)
        self.bn5 = nn.BatchNorm2d(feature_channels[3])

        # Level 4 Decoder Blocks
        self.up4 = nn.ConvTranspose2d(
            feature_channels[3], feature_channels[2], kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(
            feature_channels[2] + feature_channels[2] + group_token_channels[3],
            feature_channels[2],
            kernel_size=3,
            padding=1)
        self.bn4 = nn.BatchNorm2d(feature_channels[2])

        # Level 3 Decoder Blocks
        self.up3 = nn.ConvTranspose2d(
            feature_channels[2], feature_channels[1], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(
            feature_channels[1] + feature_channels[1] + group_token_channels[2],
            feature_channels[1],
            kernel_size=3,
            padding=1)
        self.bn3 = nn.BatchNorm2d(feature_channels[1])

        # Level 2 Decoder Blocks
        self.up2 = nn.ConvTranspose2d(
            feature_channels[1], feature_channels[0], kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            feature_channels[0] + feature_channels[0] + group_token_channels[1],
            feature_channels[0],
            kernel_size=3,
            padding=1)
        self.bn2 = nn.BatchNorm2d(feature_channels[0])

        # Level 1 Decoder Blocks
        self.up1 = nn.ConvTranspose2d(
            feature_channels[0], feature_channels[0], kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            feature_channels[0] + feature_channels[0] + group_token_channels[0],
            feature_channels[0],
            kernel_size=3,
            padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channels[0])

        # Final Convolution
        self.final_conv = nn.Conv2d(feature_channels[0], num_classes, kernel_size=1)

    def compute_hw(self, N):
        # Compute H and W from the number of tokens N
        for H in range(int(N ** 0.5), 0, -1):
            if N % H == 0:
                W = N // H
                return H, W
        return N, 1  # If N is prime

    def forward(self, features, group_tokens_list):
        # Unpack features and their spatial dimensions
        (f0, hw0), (f1, hw1), (f2, hw2), (f3, hw3), (f4, hw4) = features
        # Reverse group tokens list to align with decoder levels
        group_tokens_list = group_tokens_list[::-1]  # From deepest to shallowest

        # Level 5
        x = f4.permute(0, 2, 1).contiguous().view(-1, f4.shape[2], *hw4)
        x = self.up5(x)
        f3 = f3.permute(0, 2, 1).contiguous().view(-1, f3.shape[2], *hw3)
        if f3.shape[2:] != x.shape[2:]:
            f3 = F.interpolate(f3, size=x.shape[2:], mode='bilinear', align_corners=False)

        group_token_features = group_tokens_list[0]  # [B, N, C]
        B, N, C = group_token_features.shape
        H, W = self.compute_hw(N)
        group_token_features = group_token_features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        group_token_features = F.interpolate(group_token_features, size=x.shape[2:], mode='nearest')

        x = torch.cat([x, f3, group_token_features], dim=1)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        # Level 4
        x = self.up4(x)
        f2 = f2.permute(0, 2, 1).contiguous().view(-1, f2.shape[2], *hw2)
        if f2.shape[2:] != x.shape[2:]:
            f2 = F.interpolate(f2, size=x.shape[2:], mode='bilinear', align_corners=False)

        group_token_features = group_tokens_list[1]
        B, N, C = group_token_features.shape
        H, W = self.compute_hw(N)
        group_token_features = group_token_features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        group_token_features = F.interpolate(group_token_features, size=x.shape[2:], mode='nearest')

        x = torch.cat([x, f2, group_token_features], dim=1)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # Level 3
        x = self.up3(x)
        f1 = f1.permute(0, 2, 1).contiguous().view(-1, f1.shape[2], *hw1)
        if f1.shape[2:] != x.shape[2:]:
            f1 = F.interpolate(f1, size=x.shape[2:], mode='bilinear', align_corners=False)

        group_token_features = group_tokens_list[2]
        B, N, C = group_token_features.shape
        H, W = self.compute_hw(N)
        group_token_features = group_token_features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        group_token_features = F.interpolate(group_token_features, size=x.shape[2:], mode='nearest')

        x = torch.cat([x, f1, group_token_features], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # Level 2
        x = self.up2(x)
        f0 = f0.permute(0, 2, 1).contiguous().view(-1, f0.shape[2], *hw0)
        if f0.shape[2:] != x.shape[2:]:
            f0 = F.interpolate(f0, size=x.shape[2:], mode='bilinear', align_corners=False)

        group_token_features = group_tokens_list[3]
        B, N, C = group_token_features.shape
        H, W = self.compute_hw(N)
        group_token_features = group_token_features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        group_token_features = F.interpolate(group_token_features, size=x.shape[2:], mode='nearest')

        x = torch.cat([x, f0, group_token_features], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Level 1
        x = self.up1(x)
        # Since we don't have a lower-level feature, we'll use f0 again or zeros
        if f0.shape[2:] != x.shape[2:]:
            f0 = F.interpolate(f0, size=x.shape[2:], mode='bilinear', align_corners=False)

        group_token_features = group_tokens_list[4]
        B, N, C = group_token_features.shape
        H, W = self.compute_hw(N)
        group_token_features = group_token_features.permute(0, 2, 1).contiguous().view(B, C, H, W)
        group_token_features = F.interpolate(group_token_features, size=x.shape[2:], mode='nearest')

        x = torch.cat([x, f0, group_token_features], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Final upsampling to match the input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.final_conv(x)
        return x


class GroupViTSegmenter(nn.Module):
    def __init__(self, num_classes, img_size=224, **kwargs):
        super(GroupViTSegmenter, self).__init__()
        self.encoder = GroupViTEncoder(img_size=img_size, **kwargs)

        # Collect feature channels from the encoder
        feature_channels = self.encoder.feature_dims  # Length should be 5

        # Define group_token_channels based on num_group_tokens and embedding dimensions
        num_group_tokens = self.encoder.num_group_tokens  # Length should be 5
        embed_factors = kwargs.get('embed_factors', [1, 1, 1, 1, 1])
        group_token_channels = [int(self.encoder.embed_dim * embed_factors[i]) for i in range(len(num_group_tokens))]

        self.decoder = UNetDecoder(
            feature_channels=feature_channels,
            group_token_channels=group_token_channels,
            num_classes=num_classes
        )

        # Optionally, remove the group_token_classifier if not used
        # self.group_token_classifier = nn.Linear(self.encoder.num_features, num_classes)

    def forward(self, x):
        # Obtain features and group tokens from the encoder
        x, group_tokens_list, features = self.encoder(x, return_group_tokens=True)

        # Optionally, use the final group tokens for classification
        # Pass features and group tokens to the decoder
        out = self.decoder(features, group_tokens_list)

        # Return only the segmentation output
        return out
