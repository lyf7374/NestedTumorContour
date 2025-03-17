import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .CKD_model import Img_CDK_encoder
from .Point_PN import EncP
import torch.utils.checkpoint as checkpoint

def initialize_weights(model):
    """
    Initializes model weights to prevent large initial outputs that can cause numerical instability.
    
    Args:
        model (nn.Module): The neural network model.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class DecoderLayer(nn.Module):
    """
    One layer of a Transformer decoder:
      - Self-attn among contour tokens
      - Cross-attn with image tokens
      - Feed-forward
      - Residual + LayerNorm
    """
    def __init__(self, hidden_dim=256, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=0.1)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, 4*hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*hidden_dim, hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, contour_tokens, image_tokens):
        # Self-attn among contours
        attn_out, _ = self.self_attn(contour_tokens, contour_tokens, contour_tokens)
        contour_tokens = self.norm1(contour_tokens + attn_out)
        
        # Cross-attn with image tokens
        cross_out, _ = self.cross_attn(contour_tokens, image_tokens, image_tokens)
        contour_tokens = self.norm2(contour_tokens + cross_out)
        
        # Feed-Forward Network
        ff_out = self.ff(contour_tokens)
        contour_tokens = self.norm3(contour_tokens + ff_out)
        return contour_tokens
    

class ContourEncoder(nn.Module):
    """
    Evaluation Network (ENet) that takes two contour embeddings and an image embedding,
    and predicts the probability that g_i is better than g_j.

    Inputs:
        image_emb: (B, C, D, H, W)
        contour_emb_i: (B, contour_dim)
        contour_emb_j: (B, contour_dim)
    Output:
        prob: (B,)
    """
    def __init__(self,
                 img_channels=1024,
                 contour_dim=1152,
                 hidden_dim=256,
                 n_heads=4,
                 num_decoder_layers=1):
        super().__init__()
        self.img_proj = nn.Linear(img_channels, hidden_dim)
        self.contour_proj = nn.Linear(contour_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim=hidden_dim, n_heads=n_heads)
            for _ in range(num_decoder_layers)
        ])
        

    def forward(self, image_emb, contour_emb):
        """
        image_emb: (B, 1024, 4, 4, 4)
        contour_emb_i: (B, 1152)
        contour_emb_j: (B, 1152)
        => d_score: (B,)
        """
        B, C, D, H, W = image_emb.shape
        # Flatten image embeddings => (B, 64, 1024)

        img_tokens = image_emb.view(B, C, D * H * W).transpose(1, 2)  # (B, 64, 1024)

        img_tokens = self.img_proj(img_tokens)                        # (B, 64, hidden_dim)
        
        # Project contour embeddings => (B, 2, hidden_dim)
        contour_tokens = self.contour_proj(contour_emb).unsqueeze(1)  # Now (B, 1, hidden_dim)

        
        # Pass through Transformer decoder layers
        for layer in self.layers:
            contour_tokens = layer(contour_tokens, img_tokens)            # (B, 1, hidden_dim)
        
        # final embedding for contour
        e = contour_tokens.squeeze(1)  # (B, hidden_dim)
        e_norm = F.normalize(e, p=2, dim=-1)
        return e_norm

class CombinedModel(nn.Module):
    """
    Combines:
      - Parametric PC Encoder (EncP)
      - 3D Image Encoder (Img_CDK_encoder)
      - ContourEncoder for cross-attention
    """
    def __init__(self, pc_encoder, img_encoder, contour_encoder):
        super().__init__()
        self.pc_encoder = pc_encoder
        self.img_encoder = img_encoder
        self.contour_encoder = contour_encoder

    def forward(self, pc_data, images):
        """
        Args:
          pc_data:   shape [B, 3, N] (or [B, N, 3])  -> your EncP expects [B, xyz_channels, N] in forward
          images:    shape [B, 4, D, H, W] or [1, 4, D, H, W]
                     - If shape[0] = 1 but pc_data.shape[0] = B>1, we expand the image.

        Returns:
          final embedding of shape [B, hidden_dim] from ContourEncoder
        """
        B_pc = pc_data.shape[0]
        B_img = images.shape[0]

        # If the image batch size is 1 but we have multiple point clouds,
        # expand the image in memory (no actual copy) to match pc batch size

        img_emb = self.img_encoder(images)
        img_emb  = img_emb.expand(B_pc, -1, -1, -1, -1)
     
        # 1) Parametric PC encoder => [B, 1152] (in your config)
        #    Make sure xyz and feats are set appropriately for EncP
        xyz = pc_data.permute(0, 2, 1)   # [B, N, 3]
        feats = pc_data                 # if you only have (x, y, z), feats == pc_data
        pc_emb = self.pc_encoder(xyz, feats)

        # 2) 3D Image encoder => [B, 1024, D', H', W'] in your config
       

        # 3) ContourEncoder => final embedding [B, hidden_dim]
        final_emb = self.contour_encoder(img_emb, pc_emb)
        return final_emb


def get_image_and_contours_rank(data,device=None):
    """
    Convert the necessary parts of the data dictionary into PyTorch tensors
    and move them to the specified device on-demand.
    
    Assumptions:
      - data['img_embedding'] is a numpy array of shape (1024,4,4,4)
      - data['cts'] contains sub-dictionaries for 'TC_s', 'WT_l', 'TC_l', etc.
        with 'item_1' (scores) and 'item_2' (contour embeddings).
    
    This function extracts a subset of indices (using your specific logic)
    and combines the contour information into tensors.
    """
    
    patient_image =np.stack([
        data['img']['t1'],
        data['img']['t1gd'],
        data['img']['t2'],
        data['img']['flair_data']
    ], axis=0, dtype=np.float32)

    patient_image = torch.tensor(patient_image, dtype=torch.float32, device=device)
    
    # Define indices (this example uses your previous approach)
    indices = np.concatenate([np.arange(i + 25, i + 50) for i in range(0, 200, 50)])

    # Helper function to convert arrays to tensors and move to device
    def to_tensor(x):
        if not isinstance(x, torch.Tensor):
            return torch.tensor(x, dtype=torch.float32, device=device)
        return x.to(device)

    # Process contour information from different keys.
    # (Make sure that the keys 'TC_s', 'WT_l', 'TC_l' exist in your HDF5 file.)
    dis_tcs = to_tensor(data['cts']['TC_s']['item_1'])[:, indices].squeeze(0)
    g_tcs = to_tensor(data['cts']['TC_s']['item_0']).flip(1)[:, indices].squeeze(0)
    dis_wtl = to_tensor(data['cts']['WT_l']['item_1'])[:, indices].squeeze(0)
    g_wtl = to_tensor(data['cts']['WT_l']['item_0'])[:, indices].squeeze(0)
    dis_tcl = to_tensor(data['cts']['TC_l']['item_1']).squeeze(0)
    g_tcl = to_tensor(data['cts']['TC_l']['item_0']).squeeze(0)

    # Combine the scores and embeddings along the appropriate dimension
    all_scores_t = torch.cat([dis_tcs, dis_tcl, dis_wtl], dim=0)
    all_g_t = torch.cat([g_tcs, g_tcl, g_wtl], dim=0)

    '''
    torch.Size([4,  128, 128, 128]),torch.Size([500, 4096, 3]),torch.Size([500])
    '''
    return patient_image.squeeze(1), all_g_t, all_scores_t




def all_pair_indices(n):
    """
    Returns two 1D tensors i_idx, j_idx for all pairs (i<j).
    For n=4 => (0,1), (0,2), (0,3), (1,2), (1,3), (2,3).
    """
    i_list, j_list = [], []
    for i in range(n):
        for j in range(i+1, n):
       
            i_list.append(i)
            j_list.append(j)
    return torch.tensor(i_list), torch.tensor(j_list)

def infiltration_loss_multi(
    e_inf: torch.Tensor,   # (N_inf, dim) embeddings for the infiltration region
    d_inf: torch.Tensor,   # (N_inf,) each in (0,1), infiltration "distance" or "score"
    e_tc:  torch.Tensor,   # (N_tc, dim) embeddings for tumor core
    e_ht:  torch.Tensor,   # (N_ht, dim) embeddings for healthy
    tau:   float   = 1.0,
    w_func = None,
    detail=False
) -> torch.Tensor:
    """
    Implements infiltration constraints:
      - Compare infiltration embeddings vs. tumor embeddings => (1,d) => target = exp(-tau * (1-d))
      - Compare infiltration embeddings vs. healthy embeddings => (0,d) => target = exp(-tau * d)

    We average the cosine similarity across all e_tc (resp. e_ht) for each infiltration contour,
    then apply an MSE penalty to match the desired exponential target.

    Args:
      e_inf: (N_inf, dim), infiltration embeddings
      d_inf: (N_inf,) infiltration scores in [0,1]
      e_tc:  (N_tc, dim), tumor-core embeddings
      e_ht:  (N_ht, dim), healthy embeddings
      tau:   float, controls the exponential decay
      w_func: optional weighting function w(d), e.g. downweight near d=0.5

    Returns:
      Scalar infiltration loss.
    """
    device = e_inf.device
    if w_func is None:
        # default: weight=1 for all infiltration distances
        w_func = lambda d: torch.ones_like(d, device=device)

    # 1) Average infiltration->tumor similarity for each infiltration contour
    #    shape (N_inf, N_tc) => average over axis=1 => (N_inf,)
    #    We'll normalize embeddings so cos() is direct dot-product.
    e_inf_norm = F.normalize(e_inf, p=2, dim=-1)
    e_tc_norm  = F.normalize(e_tc,  p=2, dim=-1)
    e_ht_norm  = F.normalize(e_ht,  p=2, dim=-1)

    sim_tc_matrix = torch.matmul(e_inf_norm, e_tc_norm.transpose(0,1))  # (N_inf, N_tc)
    sim_tc_avg    = sim_tc_matrix.mean(dim=1)                           # (N_inf,)

    # 2) target for infiltration->tumor is exp(-tau*(1 - d_inf))
    target_tc = torch.exp(-tau * (1.0 - d_inf))  # (N_inf,)
    w_tc = w_func(d_inf)                         # (N_inf,)
    loss_tc = w_tc * (sim_tc_avg - target_tc)**2

    # 3) infiltration->healthy similarity
    #    shape (N_inf, N_ht) => average => (N_inf,)
    sim_ht_matrix = torch.matmul(e_inf_norm, e_ht_norm.transpose(0,1))
    sim_ht_avg    = sim_ht_matrix.mean(dim=1)

    # 4) target for infiltration->healthy is exp(-tau * d_inf)
    target_ht = torch.exp(-tau * d_inf)
    w_ht = w_func(d_inf)
    loss_ht = w_ht * (sim_ht_avg - target_ht)**2

    # 5) final infiltration loss => average across infiltration set
    #    you can also do a sum if you prefer
    #    or combine them: e.g. a 0.5 factor if you want them equally weighted
    loss_inf = (loss_tc.mean() + loss_ht.mean()) * 0.5
    if detail:
        print(f'Loss (d,1): {loss_tc.mean().item():.5f}, (d,0): {loss_ht.mean().item():.5f}')
    return loss_inf

def infiltration_loss_pairs(
    e_inf: torch.Tensor,   # (N_inf, dim) embeddings for the infiltration region
    d_inf: torch.Tensor,   # (N_inf,) each in (0,1), infiltration "distance" or "score"
    e_tc:  torch.Tensor,   # (N_tc, dim) embeddings for tumor core
    e_ht:  torch.Tensor,   # (N_ht, dim) embeddings for healthy
    tau:   float   = 5.0,
    w_func = None,
    detail=False
) -> torch.Tensor:
    """
    Pairwise version of the infiltration loss.

    Instead of averaging infiltration->tumor similarity before MSE, this computes
    the MSE *per (inf, tumor) pair*, and similarly for (inf, healthy) pairs.

    Args:
      e_inf: (N_inf, dim), infiltration embeddings
      d_inf: (N_inf,) infiltration scores in [0,1]
      e_tc:  (N_tc, dim), tumor-core embeddings
      e_ht:  (N_ht, dim), healthy embeddings
      tau:   float, controls the exponential decay
      w_func: optional weighting function w(d), e.g. to down-weight around d=0.5
      detail: if True, prints the partial losses

    Returns:
      Tuple of two scalars: (loss_inf, loss_dd) where loss_inf is the combined infiltration loss
      for tumor and healthy comparisons, and loss_dd is the pairwise loss over infiltration distances.
    """
    device = e_inf.device
    if w_func is None:
        # default: weight=1 for all infiltration distances
        w_func = lambda d: torch.ones_like(d, device=device)

    # 1) Normalize all embeddings (so cosine() is direct dot-product)
    e_inf_norm = F.normalize(e_inf, p=2, dim=-1)
    e_tc_norm  = F.normalize(e_tc,  p=2, dim=-1)
    e_ht_norm  = F.normalize(e_ht,  p=2, dim=-1)

    # ---------------------------------------------------------------------
    # 2) Infiltration->tumor similarities: shape (N_inf, N_tc)
    #    Target: exp(-tau * (1 - d_inf))
    # ---------------------------------------------------------------------
    sim_tc_matrix = torch.matmul(e_inf_norm, e_tc_norm.transpose(0,1))  # (N_inf, N_tc)
    target_tc_1d = torch.exp(-tau * (1.0 - d_inf))         # (N_inf,)
    target_tc_2d = target_tc_1d.unsqueeze(1).expand(-1, e_tc.size(0))  # (N_inf, N_tc)
    w_tc_1d = w_func(d_inf)  # (N_inf,)
    w_tc_2d = w_tc_1d.unsqueeze(1).expand(-1, e_tc.size(0))  # (N_inf, N_tc)
    loss_tc_mat = w_tc_2d * (sim_tc_matrix - target_tc_2d)**2
    loss_tc = loss_tc_mat.mean()

    # ---------------------------------------------------------------------
    # 3) Infiltration->healthy similarities: shape (N_inf, N_ht)
    #    Target: exp(-tau * d_inf)
    # ---------------------------------------------------------------------
    sim_ht_matrix = torch.matmul(e_inf_norm, e_ht_norm.transpose(0,1))  # (N_inf, N_ht)
    target_ht_1d = torch.exp(-tau * d_inf)  # (N_inf,)
    target_ht_2d = target_ht_1d.unsqueeze(1).expand(-1, e_ht.size(0))  # (N_inf, N_ht)
    w_ht_1d = w_func(d_inf)
    w_ht_2d = w_ht_1d.unsqueeze(1).expand(-1, e_ht.size(0))
    loss_ht_mat = w_ht_2d * (sim_ht_matrix - target_ht_2d)**2
    loss_ht = loss_ht_mat.mean()

    # ---------------------------------------------------------------------
    # 4) Infiltration-to-infiltration self-similarity loss (loss_dd)
    # ---------------------------------------------------------------------
    sim_inf_matrix = torch.matmul(e_inf_norm, e_inf_norm.transpose(0, 1))
    d_diff_matrix = torch.abs(d_inf.unsqueeze(1) - d_inf.unsqueeze(0))
    target_inf_matrix = torch.exp(-tau * d_diff_matrix)
    loss_dd = ((sim_inf_matrix - target_inf_matrix) ** 2).mean()

    # ---------------------------------------------------------------------
    # 5) Combine infiltration->tumor and infiltration->healthy
    # ---------------------------------------------------------------------
    loss_inf = 0.5 * (loss_tc + loss_ht)

    if detail:
        print(f'Loss(d,1): {loss_tc.item():.5f}, Loss(d,0): {loss_ht.item():.5f}, Loss(d,d): {loss_dd.item():.5f}')

    return loss_inf, loss_dd 


def multi_part_loss(
    emb_g,
    d_score=None,
    n_tc=100,
    n_inf=300,
    margin=0.0,
    tau=5.0,
    w_func=None,
    lambda_pos=1.0,
    lambda_neg=1.0,
    lambda_smooth=0.0,
    lambda_inf=0.0,
    lambda_dd=0.0
):
    """
    emb_g: (P, hidden_dim) the embeddings for:
       - 0..n_tc-1 : Tumor core
       - n_tc..(n_tc+n_inf-1): Infiltration zone
       - (n_tc+n_inf)..end : Healthy

    margin: used for negative pairs (TC vs HT)
    lambda_pos: weight for positive loss
    lambda_neg: weight for negative loss
    lambda_smooth: weight for infiltration adjacency smoothing
    lambda_inf: weight for infiltration loss
    lambda_dd: weight for the new infiltration difference loss

    Returns: tuple of scalar losses:
       (total_loss, pos_loss, neg_loss, loss_inf, loss_dd, smooth_loss)
    """
    device = emb_g.device
    dtype = emb_g.dtype

    # 1) Partition the embeddings
    e_tc = emb_g[:n_tc]              # (n_tc, hidden_dim)
    e_inf = emb_g[n_tc : n_tc+n_inf]   # (n_inf, hidden_dim)
    e_ht = emb_g[n_tc+n_inf :]         # (remaining, hidden_dim)

    # -----------------------------------------------------
    # 2) Positive Loss: push same-region embeddings together
    # -----------------------------------------------------
    if e_tc.size(0) > 1:
        i_idx, j_idx = all_pair_indices(e_tc.size(0))
        emb_i = e_tc[i_idx]
        emb_j = e_tc[j_idx]
        cos_ij = F.cosine_similarity(emb_i, emb_j, dim=-1)
        pos_loss_tc = ((1.0 - cos_ij) ** 2).mean()
    else:
        pos_loss_tc = torch.tensor(0.0, device=device, dtype=dtype)

    if e_ht.size(0) > 1:
        i_idx, j_idx = all_pair_indices(e_ht.size(0))
        emb_i = e_ht[i_idx]
        emb_j = e_ht[j_idx]
        cos_ij = F.cosine_similarity(emb_i, emb_j, dim=-1)
        pos_loss_ht = ((1.0 - cos_ij) ** 2).mean()
    else:
        pos_loss_ht = torch.tensor(0.0, device=device, dtype=dtype)

    pos_loss = (pos_loss_tc + pos_loss_ht) / 2

    # -----------------------------------------------------
    # 3) Negative Loss: push TC vs. HT apart if cos_ij > margin
    # -----------------------------------------------------
    if e_tc.size(0) > 0 and e_ht.size(0) > 0:
        i_idx = torch.arange(e_tc.size(0), device=device)
        j_idx = torch.arange(e_ht.size(0), device=device)
        i_idx = i_idx.unsqueeze(1).expand(-1, e_ht.size(0)).reshape(-1)
        j_idx = j_idx.unsqueeze(0).expand(e_tc.size(0), -1).reshape(-1)
        emb_i = e_tc[i_idx]
        emb_j = e_ht[j_idx]
        cos_ij = F.cosine_similarity(emb_i, emb_j, dim=-1)
        margin_diff = F.relu(cos_ij - margin)
        neg_loss = (margin_diff ** 2).mean()
    else:
        neg_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # -----------------------------------------------------
    # 4) Infiltration Loss
    # -----------------------------------------------------
    if e_inf.size(0) > 1 and lambda_inf > 0.0:
        if d_score is not None:
            d_inf = d_score[n_tc:n_tc+n_inf]
        else:
            raise ValueError("d_score must be provided if lambda_inf > 0.")
        loss_inf, loss_dd = infiltration_loss_pairs(e_inf, d_inf, e_tc, e_ht, tau, w_func)
    else:
        loss_inf = torch.tensor(0.0, device=device, dtype=dtype)
        loss_dd = torch.tensor(0.0, device=device, dtype=dtype)

    # -----------------------------------------------------
    # 5) Smoothness Loss for infiltration adjacency
    # -----------------------------------------------------
    if e_inf.size(0) > 1 and lambda_smooth > 0.0:
        sum_val = torch.tensor(0.0, device=device, dtype=dtype)
        for i in range(e_inf.size(0) - 1):
            cos_ii1 = F.cosine_similarity(e_inf[i].unsqueeze(0), e_inf[i+1].unsqueeze(0), dim=-1)
            sum_val = sum_val + (1.0 - cos_ii1) ** 2
        smooth_loss = sum_val / (e_inf.size(0) - 1)
    else:
        smooth_loss = torch.tensor(0.0, device=device, dtype=dtype)

    # -----------------------------------------------------
    # 6) Combine all losses
    # -----------------------------------------------------
    total_loss = (
        lambda_pos * pos_loss +
        lambda_neg * neg_loss +
        lambda_inf * loss_inf +
        lambda_dd * loss_dd +
        lambda_smooth * smooth_loss
    )

    return total_loss, pos_loss, neg_loss, loss_inf, loss_dd, smooth_loss




def train_contrastive_ranking_epoch_rank_order(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    # Multi-part hyperparameters
    n_tc=100,
    n_inf=300,
    margin=0.0,
    tau=5.0,
    lambda_pos=1.0,
    lambda_neg=1.0,
    lambda_smooth=0.0,
    lambda_inf=0.0,
    lambda_dd=0.0,
    w_func=None,
    eval_interval=10,
    mini_batch_size = 64 
):
    """
    Train one epoch using a combined "multi_part_loss" on a single-encoder approach.
    Also runs a test loop every 'eval_interval' epochs.

    Args:
        model, optimizer: your network and optimizer
        train_dataset, test_dataset: iterables of data items
        epoch: int, current epoch index
        n_tc: int, how many contours are tumor core
        n_inf: int, how many are infiltration
        margin: float, margin for negative pairs
        tau: float, infiltration exponential scale
        lambda_pos, lambda_neg, lambda_smooth, lambda_inf, lambda_dd: weighting for various loss terms
        w_func: optional weighting function for infiltration d-scores
        eval_interval: run the test loop if (epoch+1) % eval_interval == 0

    Returns:
        (avg_train_loss, test_loss) for this epoch
    """
    device = next(model.parameters()).device
    model.train()

    total_loss   = 0.0
    total_pos    = 0.0
    total_neg    = 0.0
    total_inf    = 0.0
    total_dd     = 0.0
    total_smooth = 0.0
    total_batches = 0

    # --------------------- TRAINING LOOP ---------------------
    for val_item in train_dataset:
        image, all_g, d_scores = get_image_and_contours_rank(val_item, device=device)
       
        if len(image.shape) == 4:
            image = image.unsqueeze(0)

        image = image.to(device)
        all_g = all_g.to(device)
        d_scores = d_scores.to(device)
        
        P = all_g.size(0)
        if P < (n_tc + n_inf + 1):
            continue

        # emb_g = model( all_g.permute(0,2,1),image)
        # Instead of a single forward pass on the entire all_g, we split it into minibatches.
        all_g_t = all_g.permute(0, 2, 1)  # shape: (500, 3, 4096)

        if not image.requires_grad:
            image.requires_grad_()
        if not all_g_t.requires_grad:
            all_g_t.requires_grad_()

        def forward_with_checkpoint(all_g_minibatch, image):
            return model(all_g_minibatch, image)

        embeddings = []
        for i in range(0, all_g_t.size(0), mini_batch_size):
            all_g_minibatch = all_g_t[i:i+mini_batch_size]
            # Use checkpointing: intermediate activations won't be stored,
            # they will be recomputed during backpropagation.
            emb_chunk = checkpoint.checkpoint(forward_with_checkpoint, all_g_minibatch, image)
            embeddings.append(emb_chunk)

        # Concatenate the outputs from each mini-batch
        emb_g = torch.cat(embeddings, dim=0)

        # Unpack all six loss values including the new dd loss
        loss_val, pos_val, neg_val, inf_val, dd_val, smooth_val = multi_part_loss(
            emb_g,
            d_score=d_scores,
            n_tc=n_tc,
            n_inf=n_inf,
            margin=margin,
            tau=tau,
            w_func=w_func,
            lambda_pos=lambda_pos,
            lambda_neg=lambda_neg,
            lambda_smooth=lambda_smooth,
            lambda_inf=lambda_inf,
            lambda_dd=lambda_dd
        )

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_loss   += loss_val
        total_pos    += pos_val
        total_neg    += neg_val
        total_inf    += inf_val
        total_dd     += dd_val
        total_smooth += smooth_val
        total_batches += 1

    avg_train_loss = total_loss / max(total_batches, 1)
    avg_pos        = total_pos / max(total_batches, 1)
    avg_neg        = total_neg / max(total_batches, 1)
    avg_inf        = total_inf / max(total_batches, 1)
    avg_dd         = total_dd / max(total_batches, 1)
    avg_smooth     = total_smooth / max(total_batches, 1)

    print(f"[Epoch {epoch+1}] Train multi-part Loss = {avg_train_loss.item():.4f} ")
    print(f"(pos={avg_pos.item():.4f}, neg={avg_neg.item():.4f}, inf={avg_inf.item():.4f}, dd={avg_dd.item():.4f}, sm={avg_smooth.item():.4f}) ")

    # --------------------- TEST / EVAL LOOP ---------------------
    test_loss = None
    if (epoch + 1) % eval_interval == 0:
        model.eval()
        total_test_loss = 0.0
        total_test_pos = 0.0
        total_test_neg = 0.0
        total_test_inf = 0.0
        total_test_dd  = 0.0
        total_test_sm  = 0.0
        test_batches = 0

        with torch.no_grad():
            for val_item in test_dataset:
                image, all_g, d_scores = get_image_and_contours_rank(val_item, device=device)
                if len(image.shape) == 4:
                    image = image.unsqueeze(0)
                image = image.to(device)
                all_g = all_g.to(device)
                d_scores = d_scores.to(device)
                # emb_g_test = model(all_g.permute(0,2,1),image)
                all_g_t = all_g.permute(0, 2, 1)  # shape: (500, 3, 4096)
                # adjust based on available memory

                embeddings = []
                for i in range(0, all_g_t.size(0), mini_batch_size):
                    all_g_minibatch = all_g_t[i:i+mini_batch_size]
                    # Use checkpointing: intermediate activations won't be stored,
                    # they will be recomputed during backpropagation.
                    emb_chunk = checkpoint.checkpoint(forward_with_checkpoint, all_g_minibatch, image)
                    embeddings.append(emb_chunk)

                # Concatenate the outputs from each mini-batch
                emb_g_test = torch.cat(embeddings, dim=0)
                loss_val_test, pos_vt, neg_vt, inf_vt, dd_vt, sm_vt = multi_part_loss(
                    emb_g_test,
                    d_score=d_scores,
                    n_tc=n_tc,
                    n_inf=n_inf,
                    margin=margin,
                    tau=tau,
                    w_func=w_func,
                    lambda_pos=lambda_pos,
                    lambda_neg=lambda_neg,
                    lambda_smooth=lambda_smooth,
                    lambda_inf=lambda_inf,
                    lambda_dd=lambda_dd
                )

                total_test_loss += loss_val_test
                total_test_pos  += pos_vt
                total_test_neg  += neg_vt
                total_test_inf  += inf_vt
                total_test_dd   += dd_vt
                total_test_sm   += sm_vt
                test_batches += 1

        test_loss = total_test_loss / max(test_batches, 1)
        avg_test_pos = total_test_pos / max(test_batches, 1)
        avg_test_neg = total_test_neg / max(test_batches, 1)
        avg_test_inf = total_test_inf / max(test_batches, 1)
        avg_test_dd  = total_test_dd / max(test_batches, 1)
        avg_test_sm  = total_test_sm / max(test_batches, 1)

        print(f"# # # Evaluation for Epoch {epoch+1} # # #")
        print(f"# Test Loss={test_loss.item():.4f}, pos-neg={(avg_test_pos.item() + avg_test_neg.item())/2:.4f}")
        print(f"# pos={avg_test_pos.item():.4f}, neg={avg_test_neg.item():.4f}, inf={avg_test_inf.item():.4f}, dd={avg_test_dd.item():.4f}, sm={avg_test_sm.item():.4f}")
    return avg_train_loss, test_loss
