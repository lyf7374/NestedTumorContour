import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time



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

def get_image_and_contours(data):
    """
    Suppose 'data' is a dictionary with keys:
       'img_embedding' = (1024,4,4,4)
       'cts' dict => 'TC_s', 'WT_l', 'TC_l', 'WT_s' each has 'item_1','item_2' 
         item_1 = (200,) array of ground truth scores
         item_2 = (200,1152) array of contour embeddings
    We'll combine them into (N,1152) and (N,) for convenience.
    """
    img_emb = data['img_embedding']

    if isinstance(img_emb, torch.Tensor):
        pass
    else:
        img_emb = torch.from_numpy(data['img_embedding']).float()  # (1024,4,4,4)
    img_emb = img_emb.squeeze(0)

    dis_tcs  = data['cts']['TC_s']['item_1'].squeeze(0)  # definitely positive => near 1
    g_emb_tcs= data['cts']['TC_s']['item_2'].squeeze(0)  
    
    dis_wtl  = data['cts']['WT_l']['item_1'].squeeze(0)  # definitely negative => near 0
    g_emb_wtl= data['cts']['WT_l']['item_2'].squeeze(0)
    
    dis_tcl  = data['cts']['TC_l']['item_1'].squeeze(0)  # uncertain => (0,1)
    g_emb_tcl= data['cts']['TC_l']['item_2'].squeeze(0)  
    
    dis_wts  = data['cts']['WT_s']['item_1'].squeeze(0)
    g_emb_wts= data['cts']['WT_s']['item_2'].squeeze(0)
    
    # Combine
    all_scores = np.concatenate([dis_tcs, dis_wtl, dis_tcl, dis_wts], axis=0)  # shape (800,)
    all_g_embs = np.concatenate([g_emb_tcs, g_emb_wtl, g_emb_tcl, g_emb_wts], axis=0) # (800,1152)
    
    all_scores_t = torch.from_numpy(all_scores).float()  # (800,)
    all_g_embs_t = torch.from_numpy(all_g_embs).float()  # (800,1152)
    
    return img_emb, all_g_embs_t, all_scores_t



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

    
    # Convert image embedding to a tensor and move to device
    img_emb = data['img_embedding']
    if not isinstance(img_emb, torch.Tensor):
        img_emb = torch.tensor(img_emb, dtype=torch.float32, device=device)
    else:
        img_emb = img_emb.to(device)
    img_emb = img_emb.squeeze(0)  # e.g., remove singleton batch dimension

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
    g_emb_tcs = to_tensor(data['cts']['TC_s']['item_2']).flip(1)[:, indices].squeeze(0)
    dis_wtl = to_tensor(data['cts']['WT_l']['item_1'])[:, indices].squeeze(0)
    g_emb_wtl = to_tensor(data['cts']['WT_l']['item_2'])[:, indices].squeeze(0)
    dis_tcl = to_tensor(data['cts']['TC_l']['item_1']).squeeze(0)
    g_emb_tcl = to_tensor(data['cts']['TC_l']['item_2']).squeeze(0)

    # Combine the scores and embeddings along the appropriate dimension
    all_scores_t = torch.cat([dis_tcs, dis_tcl, dis_wtl], dim=0)
    all_g_embs_t = torch.cat([g_emb_tcs, g_emb_tcl, g_emb_wtl], dim=0)

    return img_emb, all_g_embs_t, all_scores_t
# def get_image_and_contours_rank(data_dict, device=None):
#     # data_dict is like:
#     # {
#     #   'img_embedding': (1024,4,4,4),
#     #   'contour_embedding': (N, embed_dim),
#     #   'contour_scores': (N,)
#     # }

#     # Convert to torch Tensors on device
#     img_emb = torch.tensor(data_dict['img_embedding'], dtype=torch.float32, device=device)
#     all_g_embs = torch.tensor(data_dict['contour_embedding'], dtype=torch.float32, device=device)
#     all_scores = torch.tensor(data_dict['contour_scores'], dtype=torch.float32, device=device)

#     return img_emb, all_g_embs, all_scores

#############################################################
# Modified Transformer Decoder Model (ENet)
#############################################################

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
    

class ENet_v2(nn.Module):
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
                 num_decoder_layers=3):
        super().__init__()
        self.img_proj = nn.Linear(img_channels, hidden_dim)
        self.contour_proj = nn.Linear(contour_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_dim=hidden_dim, n_heads=n_heads)
            for _ in range(num_decoder_layers)
        ])
        
        # After processing, we'll concatenate the representations of g_i and g_j
        self.score_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image_emb, contour_emb_i, contour_emb_j):
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
        contour_emb = torch.stack([contour_emb_i, contour_emb_j], dim=1)  # (B, 2, contour_dim)
        contour_tokens = self.contour_proj(contour_emb)                   # (B, 2, hidden_dim)
        
        # Pass through Transformer decoder layers
        for layer in self.layers:
            contour_tokens = layer(contour_tokens, img_tokens)            # (B, 2, hidden_dim)
        
        # Extract the representations for g_i and g_j
        g_i_rep = contour_tokens[:, 0, :]  # (B, hidden_dim)
        g_j_rep = contour_tokens[:, 1, :]  # (B, hidden_dim)
        
        # L2 normalize the representations
        g_i_rep_norm = F.normalize(g_i_rep, p=2, dim=-1)
        g_j_rep_norm = F.normalize(g_j_rep, p=2, dim=-1)
        
        # Compute cosine similarity using the normalized representations
        sim_score = F.cosine_similarity(g_i_rep_norm, g_j_rep_norm)
        
        # Concatenate representations for additional scoring
        combined = torch.cat([g_i_rep, g_j_rep], dim=1)  # (B, 2 * hidden_dim)
        d_score = self.score_head(combined).squeeze(-1)     # (B,)
        
        return sim_score, d_score


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

def get_target_distance(si, sj):
    """
    si, sj in {0,1} or partial in (0,1).
    Return the "desired" distance.
      1) (1,1) or (0,0) => 0
      2) (1,0) or (0,1) => 1
      3) (1,d) => (1 - d)
      4) (0,d) => d
      5) (d1,d2) => |d1 - d2|
    """
    # both absolute same
    if (si == 0.0 and sj == 0.0) or (si == 1.0 and sj == 1.0):
        return 0.0
    # both absolute mismatch
    if (si == 0.0 and sj == 1.0) or (si == 1.0 and sj == 0.0):
        return 1.0
    
    # one absolute, one partial
    if si == 1.0 and (0 < sj < 1.0):
        return 1.0 - sj
    if sj == 1.0 and (0 < si < 1.0):
        return 1.0 - si
    if si == 0.0 and (0 < sj < 1.0):
        return sj
    if sj == 0.0 and (0 < si < 1.0):
        return si
    
    # both partial => difference
    return abs(si - sj)

def sample_pairs_no_diagonal(N, num_pairs, start=None, device=None):
    """
    Samples 'num_pairs' valid (i, j) with i != j from range(N).
    Returns tensors (i_idx, j_idx).
    """
    
    all_idx = torch.arange(N * N, device=device)


    # 2. Build a mask for diagonal entries (where i_idx == j_idx)
    #    i == j if floor_div == modulo
    mask_diagonal = (all_idx // N) == (all_idx % N)

    # 3. Filter out diagonal indices
    valid_idx = all_idx[~mask_diagonal]

    # 4. Randomly sample from valid_idx
    chosen = valid_idx[torch.randint(0, valid_idx.shape[0], (num_pairs,), device=device)]

    # 5. Convert back to (i_idx, j_idx)
    i_idx = chosen // N
    j_idx = chosen % N
    if start!=None:
        i_idx = i_idx + start
        j_idx = j_idx + start
    return i_idx, j_idx



def margin_based_loss(sim_score, mask_A, mask_B, certrain_mask, margin=0.0):
    """
    sim_score: Tensor of cosine similarities, shape (B,)
    mask_A: Boolean tensor for positive pairs, shape (B,)
    mask_B: Boolean tensor for negative pairs, shape (B,)
    certrain_mask: Weighting mask computed from lambda_A and lambda_B, shape (B,)
    margin: The margin for negative pairs.
    
    This function computes a loss that encourages:
      - sim_score to be close to 1 for positive pairs (mask_A)
      - sim_score to be below the margin for negative pairs (mask_B)
      
    Only pairs that are in mask_A or mask_B are considered.
    """
    device = sim_score.device
    # Create a combined mask for only the certain pairs (mask_A or mask_B)
    combined_mask = mask_A | mask_B
    if combined_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    # Select only the indices where mask_A or mask_B is True
    sim_score_sel = sim_score[combined_mask]
    certrain_mask_sel = certrain_mask[combined_mask]
    mask_A_sel = mask_A[combined_mask]
    # For negative pairs, we don't need to reassign since default is 0

    # Create target labels: 1 for positive pairs (mask_A_sel), 0 for negative pairs.
    target = torch.zeros_like(sim_score_sel)
    target[mask_A_sel] = 1.0

    # Compute losses for positive and negative pairs
    pos_loss = target * (1.0 - sim_score_sel) ** 2
    neg_loss = (1.0 - target) * F.relu(sim_score_sel - margin) ** 2
    loss = (pos_loss + neg_loss) * certrain_mask_sel

    return loss.mean()

  
def soft_gaussian_loss(sim, d1, d2, sigma=0.05):
    """
    Weighted MSE to push sim->1 only if |d1 - d2| is very small.
    Uses a Gaussian weighting around 0 difference.
    """
    diff = torch.abs(d1 - d2)
    w = torch.exp(-(diff**2)/(2*sigma*sigma))  # shape (P,)
    loss_vals = w * (sim - 1.0)**2
    return loss_vals.mean()





def weight_func(d, center=0.5, scale=4.0):
    """
    Example weighting: w(d) = 1 - scale * (d - center)^2
    Clamped at >= 0. This penalizes pairs less near d=0.5.
    """
    w = 1.0 - scale * (d - center)**2
    return w.clamp_min(0.0)

def infiltration_loss(
    sim_score: torch.Tensor, 
    score_i: torch.Tensor, 
    score_j: torch.Tensor,
    mask_C: torch.Tensor,  # (1,d) or (d,1)
    mask_D: torch.Tensor,  # (0,d) or (d,0)
    mask_E: torch.Tensor,  # (d1,d2)
    tau: float = 1.0,
    w_func = None,
    include_mask_E: bool = False
):
    """
    Compute the MSE-like infiltration losses for uncertain region pairs:
      - L_{d,1} when exactly one is 1.0, the other is d in (0,1)
      - L_{d,0} when exactly one is 0.0, the other is d in (0,1)
      - L_{d1,d2} when both are in (0,1) (optional, controlled by include_mask_E)
    
    sim_score: (P,) predicted similarity for each pair.
    score_i, score_j: (P,) infiltration 'scores' (0, 1, or in-between).
    mask_C, mask_D, mask_E: (P,) boolean masks for each infiltration scenario.
    tau: float, exponential decay factor.
    w_func: function that takes d (or d1, d2) and returns weight w.
    include_mask_E: if True, include loss for mask_E; otherwise, ignore it.
    
    Returns:
        infiltration_loss: a scalar tensor averaged only over the pairs 
                           for which the respective masks are True.
    """
    device = sim_score.device
    loss_val = torch.zeros_like(sim_score, device=device)
    
    # If no weighting function is provided, define a trivial w(d)=1.
    if w_func is None:
        w_func = lambda d: torch.ones_like(d, device=device)
    
    # -------- (1,d) or (d,1) => L_{d,1} = w(d) * (s - exp(-tau*(1-d)))^2 -------
    idx_c = torch.nonzero(mask_C).squeeze()
    if idx_c.numel() > 0:
        sc_i_c = score_i[idx_c]
        sc_j_c = score_j[idx_c]
        sim_c  = sim_score[idx_c]
        
        # Identify which element is '1' vs. 'd'
        # If sc_i_c == 1, then d = sc_j_c; otherwise d = sc_i_c.
        d_c = torch.where(sc_i_c == 1.0, sc_j_c, sc_i_c)
        
        w_c = w_func(d_c)
        target_c = torch.exp(-tau * (1.0 - d_c))  # exp(-tau*(1-d))
        loss_c = w_c * (sim_c - target_c) ** 2
        
        loss_val[idx_c] = loss_c
    
    # -------- (0,d) or (d,0) => L_{d,0} = w(d) * (s - exp(-tau*d))^2 ----------
    idx_d = torch.nonzero(mask_D).squeeze()
    if idx_d.numel() > 0:
        sc_i_d = score_i[idx_d]
        sc_j_d = score_j[idx_d]
        sim_d  = sim_score[idx_d]
        
        # If sc_i_d == 0, then d = sc_j_d; otherwise d = sc_i_d.
        d_d = torch.where(sc_i_d == 0.0, sc_j_d, sc_i_d)
        
        w_d = w_func(d_d)
        target_d = torch.exp(-tau * d_d)  # exp(-tau*d)
        loss_d = w_d * (sim_d - target_d) ** 2
        
        loss_val[idx_d] = loss_d
    
    # -------- (d1,d2) => L_{d1,d2} = min(w(d1), w(d2))*(s - exp(-tau*|d1-d2|))^2 ---
    if include_mask_E:
        idx_e = torch.nonzero(mask_E).squeeze()
        if idx_e.numel() > 0:
            sc_i_e = score_i[idx_e]
            sc_j_e = score_j[idx_e]
            sim_e  = sim_score[idx_e]
            
            # w_ei  = w_func(sc_i_e)
            # w_ej  = w_func(sc_j_e)
            # w_min = torch.min(w_ei, w_ej)
            
            # diff_e = torch.abs(sc_i_e - sc_j_e)
            # target_e = torch.exp(-tau * diff_e)  # exp(-tau*|d1-d2|)
            
            # loss_e = w_min * (sim_e - target_e) ** 2
            # loss_val[idx_e] = loss_e

            loss_e = soft_gaussian_loss(sim_e, sc_i_e , sc_j_e)
            loss_val[idx_e] = loss_e
    # Create a combined mask over which to average.
    if include_mask_E:
        combined_mask = mask_C | mask_D | mask_E
    else:
        combined_mask = mask_C | mask_D

    if combined_mask.sum() > 0:
        return loss_val[combined_mask].mean()
    else:
        return torch.tensor(0.0, device=device)

def train_contrastive_ranking_epoch(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    lr=1e-4,
    num_pairs=512,
    # Hyperparameters to optionally weight similarity loss by pair type:
    alpha=0.5, 
    lambda_A=1.0,      # (1,1) or (0,0) – no sim loss
    lambda_B=1.0,      # (1,0) or (0,1)
    lambda_C=0.0,      # (1,d) or (d,1)
    lambda_D=0.0,      # (0,d) or (d,0)
    lambda_E=0.0,      # (d1,d2)
    margin=0.0,
    tau=1.0,
    w_func = False
):
    """
    Train one epoch using a combined contrastive (MSE) loss and an auxiliary similarity loss.
    
    For each data item:
      - Sample random pairs of contours.
      - Compute the predicted distance (dist_pred) via the model.
      - Compute the target distance using get_target_distance.
      - Compute an MSE loss between dist_pred and the target distance.
      - Also compute a similarity loss (derived from margin-based loss and infiltration loss)
        for each pair.
      - The similarity loss is weighted by a per-sample lambda that depends on the pair type:
          * Type A: (1,1) or (0,0) -> λ = lambda_A
          * Type B: (1,0) or (0,1) -> λ = lambda_B
          * Type C: (1,d) or (d,1) -> λ = lambda_C
          * Type D: (0,d) or (d,0) -> λ = lambda_D
          * Type E: (d1,d2)         -> λ = lambda_E
      - The final loss is: loss_total = MSE_loss + (weighted average similarity loss).
    
    Returns:
        (avg_train_loss, test_loss) for the epoch.
    """
    device = next(model.parameters()).device

    start_time = time.time()
    model.train()

    total_loss = 0.0
    total_batches = 0
    if lambda_E>0:
        include_mask_E = True
    else:
        include_mask_E = False
    W_total = lambda_A + lambda_B + lambda_C + lambda_D + lambda_E
    W_certain = (lambda_A + lambda_B) / W_total 
    W_uncertain = (lambda_C + lambda_D + lambda_E) / W_total 
    if w_func==True:
        w_func = weight_func
    else:
        w_func = None

    for data_item in train_dataset:
        # 1) Load data
        image_emb, all_g_embs, all_scores = get_image_and_contours(data_item)
        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)  # (1,1024,4,4,4)
        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)
        N = all_scores.shape[0]
        if N < 2:
            continue

        # 2) Sample random pairs
        i_idx, j_idx = sample_pairs_no_diagonal(N, num_pairs, device=device)
        # 3) Get scores for each pair (used later to build masks)
        score_i = all_scores[i_idx]
        score_j = all_scores[j_idx]

        # 4) Get contour embeddings for each pair
        g_i = all_g_embs[i_idx]
        g_j = all_g_embs[j_idx]

        # Repeat image embedding for each pair: (P,1024,4,4,4)
        image_batch = image_emb.repeat(g_i.size(0), 1, 1, 1, 1)

        # 5) Forward pass: model returns predicted distance (in [0,1])
        sim_score, emb_score = model(image_batch, g_i, g_j)  # (P,)

        # 6) Build masks for each similarity type:
        # Type A: (1,1) or (0,0)
        mask_A = ((score_i == 0.0) & (score_j == 0.0)) | ((score_i == 1.0) & (score_j == 1.0))
        # Type B: (1,0) or (0,1)
        mask_B = ((score_i == 1.0) & (score_j == 0.0)) | ((score_i == 0.0) & (score_j == 1.0))
        # Type C: one absolute and one partial: (1,d) or (d,1)
        mask_C = ((score_i == 1.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                 ((score_j == 1.0) & (score_i > 0.0) & (score_i < 1.0))
        # Type D: one absolute and one partial: (0,d) or (d,0)
        mask_D = ((score_i == 0.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                 ((score_j == 0.0) & (score_i > 0.0) & (score_i < 1.0))
        # Type E: both partial (in (0,1))
        mask_E = ((score_i > 0.0) & (score_i < 1.0) & (score_j > 0.0) & (score_j < 1.0))

        certain_mask = torch.zeros_like(score_i, dtype=torch.float32)
        
        certain_mask[mask_A] = lambda_A 
        certain_mask[mask_B] = lambda_B 

        # 7) Compute target distances (contrastive/regression target)
        dist_targets = []
        si_np = score_i.detach().cpu().numpy()
        sj_np = score_j.detach().cpu().numpy()
        for s_i, s_j in zip(si_np, sj_np):
            dist_targets.append(get_target_distance(s_i, s_j))
        dist_targets = torch.tensor(dist_targets, device=device, dtype=torch.float32)
        
        sim_loss_certain = margin_based_loss(sim_score, mask_A, mask_B, certain_mask, margin=margin)
        mse_loss_certain = F.mse_loss(emb_score[certain_mask.bool()], dist_targets[certain_mask.bool()])

        if lambda_C + lambda_D + lambda_E > 0:

            uncertain_mask = torch.zeros_like(score_i, dtype=torch.float32)
            uncertain_mask[mask_C] = lambda_C 
            uncertain_mask[mask_D] = lambda_D 
            uncertain_mask[mask_E] = lambda_E 

            sim_loss_uncertain = infiltration_loss(emb_score, score_i, score_j, mask_C, mask_D, mask_E, tau=tau, include_mask_E=include_mask_E, w_func= w_func)
            if uncertain_mask.bool().sum() > 0:
                mse_loss_uncertain = F.mse_loss(emb_score[uncertain_mask.bool()], dist_targets[uncertain_mask.bool()])
            else:
                mse_loss_uncertain = 0.0
        else:
            sim_loss_uncertain = 0.0
            mse_loss_uncertain = 0.0
        # 8) Combine losses (weighted average for certain and uncertain parts)
        loss_val = (1 - alpha) * (W_certain * mse_loss_certain + W_uncertain * mse_loss_uncertain)  \
                    + alpha * (W_certain * sim_loss_certain + W_uncertain * sim_loss_uncertain)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()

        total_batches += 1
  
    avg_train_loss = total_loss / max(total_batches, 1)
    print(f"[Epoch {epoch+1}] Train Combined Loss = {avg_train_loss:.6f} | MSE_c {mse_loss_certain:.6f} | SIM_c  {sim_loss_certain:.6f}| MSE_u {mse_loss_uncertain:.6f} | SIM_u  {sim_loss_uncertain:.6f}")
    
    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        test_batches = 0
        # Separate accumulators for certain and uncertain test losses:
        sim_record_certain = 0.0
        mse_record_certain = 0.0
        sim_record_uncertain = 0.0
        mse_record_uncertain = 0.0
        
        for _ in range(3):
            with torch.no_grad():
                for val_item in test_dataset:
                    image_emb, all_g_embs, all_scores = get_image_and_contours(val_item)
                    if len(image_emb.shape) == 4:
                        image_emb = image_emb.unsqueeze(0)
                    image_emb = image_emb.to(device)
                    all_g_embs = all_g_embs.to(device)
                    all_scores = all_scores.to(device)
                    N_test = all_scores.shape[0]
                    if N_test < 2:
                        continue
                    
                    # Sample test pairs
                    i_idx_test, j_idx_test = sample_pairs_no_diagonal(N_test, num_pairs, device=device)
                    score_i_test = all_scores[i_idx_test]
                    score_j_test = all_scores[j_idx_test]
                    
                    # Compute target distances for test pairs
                    dist_targets_test = []
                    for s_i, s_j in zip(score_i_test.detach().cpu().numpy(), score_j_test.detach().cpu().numpy()):
                        dist_targets_test.append(get_target_distance(s_i, s_j))
                    dist_targets_test = torch.tensor(dist_targets_test, device=device, dtype=torch.float32)
                    
                    # Get contour embeddings for test pairs
                    g_i_test = all_g_embs[i_idx_test]
                    g_j_test = all_g_embs[j_idx_test]
                    image_batch_test = image_emb.repeat(g_i_test.size(0), 1, 1, 1, 1)
                    
                    # Forward pass on test batch
                    sim_score_test, emb_score_test = model(image_batch_test, g_i_test, g_j_test)
                    
                    # Build masks for similarity types:
                    mask_A_test = ((score_i_test == 0.0) & (score_j_test == 0.0)) | ((score_i_test == 1.0) & (score_j_test == 1.0))
                    mask_B_test = ((score_i_test == 1.0) & (score_j_test == 0.0)) | ((score_i_test == 0.0) & (score_j_test == 1.0))
                    mask_C_test = ((score_i_test == 1.0) & (score_j_test > 0.0) & (score_j_test < 1.0)) | \
                                    ((score_j_test == 1.0) & (score_i_test > 0.0) & (score_i_test < 1.0))
                    mask_D_test = ((score_i_test == 0.0) & (score_j_test > 0.0) & (score_j_test < 1.0)) | \
                                    ((score_j_test == 0.0) & (score_i_test > 0.0) & (score_i_test < 1.0))
                    mask_E_test = ((score_i_test > 0.0) & (score_i_test < 1.0) & (score_j_test > 0.0) & (score_j_test < 1.0))
                    
                    # Certain pairs:
                    certain_mask = torch.zeros_like(score_i_test, dtype=torch.float32)
                    certain_mask[mask_A_test] = lambda_A 
                    certain_mask[mask_B_test] = lambda_B 

                    sim_loss_certain = margin_based_loss(sim_score_test, mask_A_test, mask_B_test, certain_mask, margin=margin)
                    mse_loss_certain = F.mse_loss(emb_score_test[certain_mask.bool()], dist_targets_test[certain_mask.bool()])
                    
                    # Accumulate certain losses:
                    sim_record_certain += sim_loss_certain.item()
                    mse_record_certain += mse_loss_certain.item()
                    
                    # Uncertain pairs:
                    if (lambda_C + lambda_D + lambda_E) > 0:
                        uncertain_mask = torch.zeros_like(score_i_test, dtype=torch.float32)
                        uncertain_mask[mask_C_test] = lambda_C 
                        uncertain_mask[mask_D_test] = lambda_D
                        uncertain_mask[mask_E_test] = lambda_E 
                        
                        
                        sim_loss_uncertain = infiltration_loss(emb_score_test, score_i_test, score_j_test, mask_C_test, mask_D_test, mask_E_test, tau=tau, include_mask_E=include_mask_E, w_func= w_func)
                        
                        if uncertain_mask.bool().sum() > 0:
                            mse_loss_uncertain = F.mse_loss(emb_score[uncertain_mask.bool()], dist_targets[uncertain_mask.bool()])
                        else:
                            mse_loss_uncertain = 0.0 
                    
                        sim_record_uncertain += sim_loss_uncertain.item()
                        mse_record_uncertain += mse_loss_uncertain.item()
                    else:
                        # When there are no uncertain pairs, add zero.
                        sim_record_uncertain += 0.0
                        mse_record_uncertain += 0.0
                        
                    loss_test_val = (1 - alpha) * (W_certain * mse_loss_certain + W_uncertain * mse_loss_uncertain)  \
                                        + alpha * (W_certain * sim_loss_certain + W_uncertain * sim_loss_uncertain)
                    total_test_loss += loss_test_val.item()
                    test_batches += 1

        test_loss = total_test_loss / max(test_batches, 1)
        avg_sim_certain = sim_record_certain / max(test_batches, 1)
        avg_mse_certain = mse_record_certain / max(test_batches, 1)
        avg_sim_uncertain = sim_record_uncertain / max(test_batches, 1)
        avg_mse_uncertain = mse_record_uncertain / max(test_batches, 1)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
        print(f"# # # # Test Combined Loss = {test_loss:.6f} # # # #")
        print(f"# # # # Certain: MSE = {avg_mse_certain:.6f} | SIM = {avg_sim_certain:.6f} # # # #")
        print(f"# # # # Uncertain: MSE = {avg_mse_uncertain:.6f} | SIM = {avg_sim_uncertain:.6f} # # # #")
    return avg_train_loss, test_loss



def train_contrastive_ranking_epoch_pure(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    lr=1e-4,
    num_pairs=512,
    # Hyperparameters to optionally weight similarity loss by pair type:
    alpha=1.0, 
    lambda_A=1.0,      # (1,1) or (0,0) – no sim loss
    lambda_B=1.0,      # (1,0) or (0,1)
    lambda_C=0.0,      # (1,d) or (d,1)
    lambda_D=0.0,      # (0,d) or (d,0)
    lambda_E=0.0,      # (d1,d2)
    margin=0.0,
    tau=1.0,
    w_func = False
):
    """
    Train one epoch using a combined contrastive (MSE) loss and an auxiliary similarity loss.
    
    For each data item:
      - Sample random pairs of contours.
      - Compute the predicted distance (dist_pred) via the model.
      - Compute the target distance using get_target_distance.
      - Compute an MSE loss between dist_pred and the target distance.
      - Also compute a similarity loss (derived from margin-based loss and infiltration loss)
        for each pair.
      - The similarity loss is weighted by a per-sample lambda that depends on the pair type:
          * Type A: (1,1) or (0,0) -> λ = lambda_A
          * Type B: (1,0) or (0,1) -> λ = lambda_B
          * Type C: (1,d) or (d,1) -> λ = lambda_C
          * Type D: (0,d) or (d,0) -> λ = lambda_D
          * Type E: (d1,d2)         -> λ = lambda_E
      - The final loss is: loss_total = MSE_loss + (weighted average similarity loss).
    
    Returns:
        (avg_train_loss, test_loss) for the epoch.
    """
    device = next(model.parameters()).device

    start_time = time.time()

    model.train()

    total_loss = 0.0
    total_batches = 0
    if lambda_E>0:
        include_mask_E = True
    else:
        include_mask_E = False
    W_total = lambda_A + lambda_B + lambda_C + lambda_D + lambda_E
    W_certain = (lambda_A + lambda_B) / W_total 
    W_uncertain = (lambda_C + lambda_D + lambda_E) / W_total 
    if w_func==True:
        w_func = weight_func
    else:
        w_func = None

    for data_item in train_dataset:
        # 1) Load data
        image_emb, all_g_embs, all_scores = get_image_and_contours(data_item)
        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)  # (1,1024,4,4,4)
        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)
        N = all_scores.shape[0]
        if N < 2:
            continue

        # 2) Sample random pairs
        i_idx, j_idx = sample_pairs_no_diagonal(N, num_pairs, device=device)
        # 3) Get scores for each pair (used later to build masks)
        score_i = all_scores[i_idx]
        score_j = all_scores[j_idx]

        # 4) Get contour embeddings for each pair
        g_i = all_g_embs[i_idx]
        g_j = all_g_embs[j_idx]

        # Repeat image embedding for each pair: (P,1024,4,4,4)
        image_batch = image_emb.repeat(g_i.size(0), 1, 1, 1, 1)

        # 5) Forward pass: model returns predicted distance (in [0,1])
        sim_score, emb_score = model(image_batch, g_i, g_j)  # (P,)

        # 6) Build masks for each similarity type:
        # Type A: (1,1) or (0,0)
        mask_A = ((score_i == 0.0) & (score_j == 0.0)) | ((score_i == 1.0) & (score_j == 1.0))
        # Type B: (1,0) or (0,1)
        mask_B = ((score_i == 1.0) & (score_j == 0.0)) | ((score_i == 0.0) & (score_j == 1.0))
        # Type C: one absolute and one partial: (1,d) or (d,1)
        mask_C = ((score_i == 1.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                 ((score_j == 1.0) & (score_i > 0.0) & (score_i < 1.0))
        # Type D: one absolute and one partial: (0,d) or (d,0)
        mask_D = ((score_i == 0.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                 ((score_j == 0.0) & (score_i > 0.0) & (score_i < 1.0))
        # Type E: both partial (in (0,1))
        mask_E = ((score_i > 0.0) & (score_i < 1.0) & (score_j > 0.0) & (score_j < 1.0))

        certain_mask = torch.zeros_like(score_i, dtype=torch.float32)
        
        certain_mask[mask_A] = lambda_A 
        certain_mask[mask_B] = lambda_B 

        sim_loss_certain = margin_based_loss(sim_score, mask_A, mask_B, certain_mask, margin=margin)


        if lambda_C + lambda_D + lambda_E > 0:
            uncertain_mask = torch.zeros_like(score_i, dtype=torch.float32)
            uncertain_mask[mask_C] = lambda_C 
            uncertain_mask[mask_D] = lambda_D 
            uncertain_mask[mask_E] = lambda_E 

            sim_loss_uncertain = infiltration_loss(emb_score, score_i, score_j, mask_C, mask_D, mask_E, tau=tau, include_mask_E=include_mask_E, w_func= w_func)
        else:
            sim_loss_uncertain = 0.0

        # 8) Combine losses (weighted average for certain and uncertain parts)
        loss_val =  W_certain * sim_loss_certain + W_uncertain * sim_loss_uncertain
                    

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()

        total_batches += 1
  
    avg_train_loss = total_loss / max(total_batches, 1)
    print(f"[Epoch {epoch+1}] Train Combined Loss = {avg_train_loss:.6f} | SIM_c  {sim_loss_certain:.6f} | SIM_u  {sim_loss_uncertain:.6f}")
    
    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        test_batches = 0
        # Separate accumulators for certain and uncertain test losses:
        sim_record_certain = 0.0
        mse_record_certain = 0.0
        sim_record_uncertain = 0.0
        mse_record_uncertain = 0.0
        
        for _ in range(3):
            with torch.no_grad():
                for val_item in test_dataset:
                    image_emb, all_g_embs, all_scores = get_image_and_contours(val_item)
                    if len(image_emb.shape) == 4:
                        image_emb = image_emb.unsqueeze(0)
                    image_emb = image_emb.to(device)
                    all_g_embs = all_g_embs.to(device)
                    all_scores = all_scores.to(device)
                    N_test = all_scores.shape[0]
                    if N_test < 2:
                        continue
                    
                    # Sample test pairs
                    i_idx_test, j_idx_test = sample_pairs_no_diagonal(N_test, num_pairs, device=device)
                    score_i_test = all_scores[i_idx_test]
                    score_j_test = all_scores[j_idx_test]
                    
        
                    # Get contour embeddings for test pairs
                    g_i_test = all_g_embs[i_idx_test]
                    g_j_test = all_g_embs[j_idx_test]
                    image_batch_test = image_emb.repeat(g_i_test.size(0), 1, 1, 1, 1)
                    
                    # Forward pass on test batch
                    sim_score_test, emb_score_test = model(image_batch_test, g_i_test, g_j_test)
                    
                    # Build masks for similarity types:
                    mask_A_test = ((score_i_test == 0.0) & (score_j_test == 0.0)) | ((score_i_test == 1.0) & (score_j_test == 1.0))
                    mask_B_test = ((score_i_test == 1.0) & (score_j_test == 0.0)) | ((score_i_test == 0.0) & (score_j_test == 1.0))
                    mask_C_test = ((score_i_test == 1.0) & (score_j_test > 0.0) & (score_j_test < 1.0)) | \
                                    ((score_j_test == 1.0) & (score_i_test > 0.0) & (score_i_test < 1.0))
                    mask_D_test = ((score_i_test == 0.0) & (score_j_test > 0.0) & (score_j_test < 1.0)) | \
                                    ((score_j_test == 0.0) & (score_i_test > 0.0) & (score_i_test < 1.0))
                    mask_E_test = ((score_i_test > 0.0) & (score_i_test < 1.0) & (score_j_test > 0.0) & (score_j_test < 1.0))
                    
                    # Certain pairs:
                    certain_mask = torch.zeros_like(score_i_test, dtype=torch.float32)
                    certain_mask[mask_A_test] = lambda_A 
                    certain_mask[mask_B_test] = lambda_B 

                    sim_loss_certain = margin_based_loss(sim_score_test, mask_A_test, mask_B_test, certain_mask, margin=margin)
     
                    
                    # Accumulate certain losses:
                    sim_record_certain += sim_loss_certain.item()
      
                    
                    # Uncertain pairs:
                    if (lambda_C + lambda_D + lambda_E) > 0:
                        uncertain_mask = torch.zeros_like(score_i_test, dtype=torch.float32)
                        uncertain_mask[mask_C_test] = lambda_C 
                        uncertain_mask[mask_D_test] = lambda_D
                        uncertain_mask[mask_E_test] = lambda_E 
                        
                        
                        sim_loss_uncertain = infiltration_loss(emb_score_test, score_i_test, score_j_test, mask_C_test, mask_D_test, mask_E_test, tau=tau, include_mask_E=include_mask_E, w_func= w_func)
                        sim_record_uncertain += sim_loss_uncertain.item()
                 
                    else:
                        # When there are no uncertain pairs, add zero.
                        sim_record_uncertain += 0.0
                        mse_record_uncertain += 0.0
                        
                    loss_test_val = W_certain * sim_loss_certain + W_uncertain * sim_loss_uncertain
                    total_test_loss += loss_test_val.item()
                    test_batches += 1

        test_loss = total_test_loss / max(test_batches, 1)
        avg_sim_certain = sim_record_certain / max(test_batches, 1)
        avg_sim_uncertain = sim_record_uncertain / max(test_batches, 1)
  
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
        print(f"# # # # Test Combined Loss = {test_loss:.6f} # # # #")
        print(f"# # # # Certain: SIM = {avg_sim_certain:.6f} | Uncertain: SIM = {avg_sim_uncertain:.6f}# # # #")
    return avg_train_loss, test_loss



def margin_based_loss_emb(g_i_rep_norm, g_j_rep_norm, mask_A, mask_B, certrain_mask, margin=0.0):
    """
    Computes a loss based on cosine similarity between g_i_rep_norm and g_j_rep_norm.
    Only considers indices where either mask_A or mask_B is True.
    Positive pairs (mask_A) target a cosine similarity of 1, negative pairs (mask_B) are penalized if they exceed a margin.
    """
    device = g_i_rep_norm.device
    # Ensure that all masks and weights are on the same device.
    mask_A = mask_A.to(device)
    mask_B = mask_B.to(device)
    certrain_mask = certrain_mask.to(device)
    
    # Create a combined mask for only the certain pairs.
    combined_mask = mask_A | mask_B
    if combined_mask.sum() == 0:
        return torch.tensor(0.0, device=device)
    
    sim_score = F.cosine_similarity(g_i_rep_norm, g_j_rep_norm)
    sim_score_sel = sim_score[combined_mask]
    certrain_mask_sel = certrain_mask[combined_mask]
    mask_A_sel = mask_A[combined_mask]
    
    # Create target labels: 1 for positive pairs, 0 for negative pairs.
    target = torch.zeros_like(sim_score_sel)
    target[mask_A_sel] = 1.0

    # Compute losses for positive and negative pairs.
    pos_loss = target * (1.0 - sim_score_sel) ** 2
    neg_loss = (1.0 - target) * F.relu(sim_score_sel - margin) ** 2
    loss = (pos_loss + neg_loss) * certrain_mask_sel

    return loss.mean()

def infiltration_loss_emb(
    g_i_rep_norm: torch.Tensor,
    g_j_rep_norm: torch.Tensor,
    score_i: torch.Tensor, 
    score_j: torch.Tensor,
    mask_C: torch.Tensor,  # (1,d) or (d,1)
    mask_D: torch.Tensor,  # (0,d) or (d,0)
    mask_E: torch.Tensor,  # (d1,d2)
    tau: float = 1.0,
    w_func = None,
    include_mask_E: bool = False
):
    """
    Compute the MSE-like infiltration losses for uncertain region pairs:
      - L_{d,1} when exactly one is 1.0, the other is d in (0,1)
      - L_{d,0} when exactly one is 0.0, the other is d in (0,1)
      - L_{d1,d2} when both are in (0,1) (optional, controlled by include_mask_E)
    
    This function expects that all input tensors are on the same device.
    """
    device = g_i_rep_norm.device
    # Ensure score and mask tensors are on the same device
    score_i = score_i.to(device)
    score_j = score_j.to(device)
    mask_C = mask_C.to(device)
    mask_D = mask_D.to(device)
    mask_E = mask_E.to(device)
    
    sim_score = F.cosine_similarity(g_i_rep_norm, g_j_rep_norm)
    loss_val = torch.zeros_like(sim_score, device=device)
    
    # If no weighting function is provided, define a trivial one.
    if w_func is None:
        w_func = lambda d: torch.ones_like(d, device=device)
    
    # -------- (1,d) or (d,1) case --------
    idx_c = torch.nonzero(mask_C).squeeze()
    if idx_c.numel() > 0:
        sc_i_c = score_i[idx_c]
        sc_j_c = score_j[idx_c]
        sim_c  = sim_score[idx_c]
        # Determine d: if sc_i_c==1, then d = sc_j_c, else d = sc_i_c.
        d_c = torch.where(sc_i_c == 1.0, sc_j_c, sc_i_c)
        w_c = w_func(d_c).to(device)
        target_c = torch.exp(-tau * (1.0 - d_c)).to(device)
        loss_c = w_c * (sim_c - target_c) ** 2
        loss_val[idx_c] = loss_c
    
    # -------- (0,d) or (d,0) case --------
    idx_d = torch.nonzero(mask_D).squeeze()
    if idx_d.numel() > 0:
        sc_i_d = score_i[idx_d]
        sc_j_d = score_j[idx_d]
        sim_d  = sim_score[idx_d]
        # Determine d: if sc_i_d==0, then d = sc_j_d, else d = sc_i_d.
        d_d = torch.where(sc_i_d == 0.0, sc_j_d, sc_i_d)
        w_d = w_func(d_d).to(device)
        target_d = torch.exp(-tau * d_d).to(device)
        loss_d = w_d * (sim_d - target_d) ** 2
        loss_val[idx_d] = loss_d
    
    # -------- (d1,d2) case --------
    if include_mask_E:
        idx_e = torch.nonzero(mask_E).squeeze()
        if idx_e.numel() > 0:
            sc_i_e = score_i[idx_e]
            sc_j_e = score_j[idx_e]
            sim_e  = sim_score[idx_e]

            
            diff_e = torch.abs(sc_i_e - sc_j_e)
            target_e = torch.exp(-tau * diff_e)  # exp(-tau*|d1-d2|)
            
            loss_e = (sim_e - target_e) ** 2
            loss_val[idx_e] = loss_e
            
            # loss_e = soft_gaussian_loss(sim_e, sc_i_e, sc_j_e)
            # loss_val[idx_e] = loss_e
    
    # Combine masks to average over the valid indices.
    if include_mask_E:
        combined_mask = mask_C | mask_D | mask_E
    else:
        combined_mask = mask_C | mask_D

    if combined_mask.sum() > 0:
        return loss_val[combined_mask].mean()
    else:
        return torch.tensor(0.0, device=device)



def train_contrastive_ranking_epoch_rank_discrete(
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
    w_func=None,
    eval_interval=10,
    device=None
):
    """
    Train one epoch using a combined contrastive (MSE) loss and an auxiliary similarity loss.
    
    For each data item:
      - Sample random pairs of contours.
      - Compute the predicted distance (dist_pred) via the model.
      - Compute the target distance using get_target_distance.
      - Compute an MSE loss between dist_pred and the target distance.
      - Also compute a similarity loss (derived from margin-based loss and infiltration loss)
        for each pair.
      - The similarity loss is weighted by a per-sample lambda that depends on the pair type:
          * Type A: (1,1) or (0,0) -> λ = lambda_A
          * Type B: (1,0) or (0,1) -> λ = lambda_B
          * Type C: (1,d) or (d,1) -> λ = lambda_C
          * Type D: (0,d) or (d,0) -> λ = lambda_D
          * Type E: (d1,d2)         -> λ = lambda_E
      - The final loss is: loss_total = MSE_loss + (weighted average similarity loss).
    
    Returns:
        (avg_train_loss, test_loss) for the epoch.
    """

    start_time = time.time()

    model.train()

    total_loss = 0.0
    total_batches = 0

    lambda_A = lambda_B = 2.0
    lambda_C = lambda_D = lambda_E = 1.0
    num_pairs = 2048
    if lambda_E>0:
        include_mask_E = True
    else:
        include_mask_E = False
    W_total = lambda_A + lambda_B + lambda_C + lambda_D + lambda_E
    W_certain = (lambda_A + lambda_B) / W_total 
    W_uncertain = (lambda_C + lambda_D + lambda_E) / W_total 

    if w_func==True:
        w_func = weight_func
    else:
        w_func = None

    for data_item in train_dataset:
        # 1) Load data
        image_emb, all_g_embs, all_scores = get_image_and_contours_rank(data_item,device=device)
        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)  # (1,1024,4,4,4)
        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)

        image_batch = image_emb.repeat(all_g_embs.size(0), 1, 1, 1, 1)
        emb_g = model(image_batch, all_g_embs)  # (P,)

        e_inf = emb_g[n_tc:n_tc+n_inf]
        # -----------------------------------------------------
        if lambda_smooth > 0.0:
            sum_val = torch.tensor(0.0, device=device, dtype=e_inf.dtype)
            for i in range(e_inf.size(0) - 1):
                # Adding a dimension to ensure cosine_similarity works on 2D tensors
                cos_ii1 = F.cosine_similarity(e_inf[i].unsqueeze(0), e_inf[i+1].unsqueeze(0), dim=-1)
                sum_val = sum_val + (1.0 - cos_ii1) ** 2

            smooth_loss = sum_val / (e_inf.size(0) - 1)
        else:
            smooth_loss = torch.tensor(0.0, device=device, dtype=e_inf.dtype)

        N = all_scores.shape[0]
        if N < 2:
            continue
        # 2) Sample random pairs
        i_idx, j_idx = sample_pairs_no_diagonal(N, num_pairs, device=device)
        # 3) Get scores for each pair (used later to build masks)
        score_i = all_scores[i_idx]
        score_j = all_scores[j_idx]

        # 4) Get contour embeddings for each pair
        emb_g_i = emb_g[i_idx]
        emb_g_j = emb_g[j_idx]


        # 6) Build masks for each similarity type:
        # Type A: (1,1) or (0,0)
        mask_A = ((score_i == 0.0) & (score_j == 0.0)) | ((score_i == 1.0) & (score_j == 1.0))
        # Type B: (1,0) or (0,1)
        mask_B = ((score_i == 1.0) & (score_j == 0.0)) | ((score_i == 0.0) & (score_j == 1.0))
        # Type C: one absolute and one partial: (1,d) or (d,1)
        mask_C = ((score_i == 1.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                 ((score_j == 1.0) & (score_i > 0.0) & (score_i < 1.0))
        # Type D: one absolute and one partial: (0,d) or (d,0)
        mask_D = ((score_i == 0.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                 ((score_j == 0.0) & (score_i > 0.0) & (score_i < 1.0))
        # Type E: both partial (in (0,1))
        mask_E = ((score_i > 0.0) & (score_i < 1.0) & (score_j > 0.0) & (score_j < 1.0))

        certain_mask = torch.zeros_like(score_i, dtype=torch.float32)
        
        certain_mask[mask_A] = lambda_A 
        certain_mask[mask_B] = lambda_B 

        sim_loss_certain = margin_based_loss_emb(emb_g_i,emb_g_j, mask_A, mask_B, certain_mask, margin=margin)


        if lambda_C + lambda_D + lambda_E > 0:
            uncertain_mask = torch.zeros_like(score_i, dtype=torch.float32)
            uncertain_mask[mask_C] = lambda_C 
            uncertain_mask[mask_D] = lambda_D 
            uncertain_mask[mask_E] = lambda_E 

            sim_loss_uncertain = infiltration_loss_emb(emb_g_i,emb_g_j, score_i, score_j, mask_C, mask_D, mask_E, tau=tau, include_mask_E=include_mask_E, w_func= w_func)
        else:
            sim_loss_uncertain = 0.0

        # 8) Combine losses (weighted average for certain and uncertain parts)
        loss_val =  W_certain * sim_loss_certain + W_uncertain * sim_loss_uncertain + lambda_smooth * smooth_loss
                    

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val

        total_batches += 1
  
    avg_train_loss = total_loss / max(total_batches, 1)
    print(f"[Epoch {epoch+1}] Train Combined Loss = {avg_train_loss.item():.4f} | SIM_c  {sim_loss_certain.item():.4f} | SIM_u  {sim_loss_uncertain.item():.4f}| Smooth  {smooth_loss.item():.4f}")

    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        test_batches = 0
        # Separate accumulators for certain and uncertain test losses:
        sim_record_certain = 0.0
        sim_record_uncertain = 0.0
        sim_record_smooth = 0.0

        
        for _ in range(3):
            with torch.no_grad():
                for val_item in test_dataset:
                    image_emb, all_g_embs, all_scores = get_image_and_contours_rank(val_item,device=device)
                    if len(image_emb.shape) == 4:
                        image_emb = image_emb.unsqueeze(0)  # (1,1024,4,4,4)
                    image_emb = image_emb.to(device)
                    all_g_embs = all_g_embs.to(device)
                    all_scores = all_scores.to(device)

                    image_batch = image_emb.repeat(all_g_embs.size(0), 1, 1, 1, 1)
                    emb_g = model(image_batch, all_g_embs)  # (P,)
                    
                    e_inf = emb_g[100:350]
                    # -----------------------------------------------------
                    if e_inf.size(0) > 1 and lambda_smooth > 0.0:
                        sum_val = torch.tensor(0.0, device=device, dtype=e_inf.dtype)
                        for i in range(e_inf.size(0) - 1):
                            # Adding a dimension to ensure cosine_similarity works on 2D tensors
                            cos_ii1 = F.cosine_similarity(e_inf[i].unsqueeze(0), e_inf[i+1].unsqueeze(0), dim=-1)
                            sum_val = sum_val + (1.0 - cos_ii1) ** 2
                        smooth_loss = sum_val / (e_inf.size(0) - 1)
                    else:
                        smooth_loss = torch.tensor(0.0, device=device, dtype=e_inf.dtype)
                    sim_record_smooth += smooth_loss.item()

                    N = all_scores.shape[0]
                    if N < 2:
                        continue
                    # 2) Sample random pairs
                    i_idx, j_idx = sample_pairs_no_diagonal(N, num_pairs, device=device)
                    # 3) Get scores for each pair (used later to build masks)
                    score_i = all_scores[i_idx]
                    score_j = all_scores[j_idx]

                    # 4) Get contour embeddings for each pair
                    emb_g_i = emb_g[i_idx]
                    emb_g_j = emb_g[j_idx]


                    # 6) Build masks for each similarity type:
                    # Type A: (1,1) or (0,0)
                    mask_A = ((score_i == 0.0) & (score_j == 0.0)) | ((score_i == 1.0) & (score_j == 1.0))
                    # Type B: (1,0) or (0,1)
                    mask_B = ((score_i == 1.0) & (score_j == 0.0)) | ((score_i == 0.0) & (score_j == 1.0))
                    # Type C: one absolute and one partial: (1,d) or (d,1)
                    mask_C = ((score_i == 1.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                            ((score_j == 1.0) & (score_i > 0.0) & (score_i < 1.0))
                    # Type D: one absolute and one partial: (0,d) or (d,0)
                    mask_D = ((score_i == 0.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                            ((score_j == 0.0) & (score_i > 0.0) & (score_i < 1.0))
                    # Type E: both partial (in (0,1))
                    mask_E = ((score_i > 0.0) & (score_i < 1.0) & (score_j > 0.0) & (score_j < 1.0))

                    certain_mask = torch.zeros_like(score_i, dtype=torch.float32)
                    
                    certain_mask[mask_A] = lambda_A 
                    certain_mask[mask_B] = lambda_B 

                    sim_loss_certain = margin_based_loss_emb(emb_g_i,emb_g_j, mask_A, mask_B, certain_mask, margin=margin)
                    sim_record_certain += sim_loss_certain.item()
                        

                    if lambda_C + lambda_D + lambda_E > 0:
                        uncertain_mask = torch.zeros_like(score_i, dtype=torch.float32)
                        uncertain_mask[mask_C] = lambda_C 
                        uncertain_mask[mask_D] = lambda_D 
                        uncertain_mask[mask_E] = lambda_E 

                        sim_loss_uncertain = infiltration_loss_emb(emb_g_i,emb_g_j, score_i, score_j, mask_C, mask_D, mask_E, tau=tau, include_mask_E=include_mask_E, w_func= w_func)
                    else:
                        sim_loss_uncertain = 0.0
                    sim_record_uncertain += sim_loss_uncertain.item()
                    # 8) Combine losses (weighted average for certain and uncertain parts)
                  
                    loss_test_val =  W_certain * sim_loss_certain + W_uncertain * sim_loss_uncertain + lambda_smooth * smooth_loss
                    total_test_loss += loss_test_val.item()
                    test_batches += 1

        test_loss = total_test_loss / max(test_batches, 1)
        avg_sim_certain = sim_record_certain / max(test_batches, 1)
        avg_sim_uncertain = sim_record_uncertain / max(test_batches, 1)
        avg_smooth = sim_record_smooth  / max(test_batches, 1)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
        print(f"# # # # Test Combined Loss = {test_loss:.4f} # # # #")
        print(f"# # # # Certain: SIM = {avg_sim_certain:.4f} | Uncertain: SIM = {avg_sim_uncertain:.4f}| Smooth = {avg_smooth:.4f}# # # #")
    return avg_train_loss, test_loss


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
    eval_interval=10
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

    if w_func == True:
        w_func = weight_func
    else:
        w_func = lambda d: torch.ones_like(d, device=device)

    # --------------------- TRAINING LOOP ---------------------
    for val_item in train_dataset:
        image_emb, all_g_embs, d_scores = get_image_and_contours_rank(val_item, device=device)
        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)
        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        d_scores = d_scores.to(device)

        P = all_g_embs.size(0)
        if P < (n_tc + n_inf + 1):
            continue

        image_batch = image_emb.repeat(P, 1, 1, 1, 1)
        emb_g = model(image_batch, all_g_embs)

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
                image_emb, all_g_embs, d_scores = get_image_and_contours_rank(val_item, device=device)
                if len(image_emb.shape) == 4:
                    image_emb = image_emb.unsqueeze(0)
                image_emb = image_emb.to(device)
                all_g_embs = all_g_embs.to(device)
                d_scores = d_scores.to(device)
                P_test = all_g_embs.size(0)
                if P_test < (n_tc + n_inf + 1):
                    continue

                image_batch_test = image_emb.repeat(P_test, 1, 1, 1, 1)
                emb_g_test = model(image_batch_test, all_g_embs)

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
