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

def get_image_and_contours_neighbour(data):
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
   

    dis_tcs  =  torch.flip(data['cts']['TC_s']['item_1'][:,:50], dims=[1]).squeeze(0)  # definitely positive => near 1
    g_emb_tcs=  torch.flip(data['cts']['TC_s']['item_2'][:,:50,:], dims=[1]).squeeze(0)  
    
    dis_tcl  = data['cts']['TC_l']['item_1'][:,:50].squeeze(0)  # uncertain => (0,1)
    g_emb_tcl= data['cts']['TC_l']['item_2'][:,:50,:].squeeze(0)  

    dis_wtl  = data['cts']['WT_l']['item_1'][:,:50].squeeze(0)  # definitely negative => near 0
    g_emb_wtl= data['cts']['WT_l']['item_2'][:,:50,:].squeeze(0)
    

    # dis_wts  = data['cts']['WT_s']['item_1'].squeeze(0)
    # g_emb_wts= data['cts']['WT_s']['item_2'].squeeze(0)
    
    # Combine
    all_scores = np.concatenate([dis_tcs, dis_tcl, dis_wtl], axis=0)  # shape (150,)
    all_g_embs = np.concatenate([g_emb_tcs, g_emb_tcl, g_emb_wtl], axis=0) # (150,1152)
    
    all_scores_t = torch.from_numpy(all_scores).float()  # (150,)
    all_g_embs_t = torch.from_numpy(all_g_embs).float()  # (150,1152)
    
    return img_emb, all_g_embs_t, all_scores_t

def pairwise_bce_loss(pred_scores, gt_scores, i_idx, j_idx):
    """
    Pairwise BCE ranking:
     - pred_scores: (N,)
     - gt_scores: (N,)
     - i_idx, j_idx: (num_pairs,) random pairs
       label=1 if i outranks j
    """
    dist_i = gt_scores[i_idx]
    dist_j = gt_scores[j_idx]
    s_i = pred_scores[i_idx]
    s_j = pred_scores[j_idx]

    # Now: if dist_i > dist_j, i is "more tumor" -> label=1
    labels = (dist_i > dist_j).float()
    
    delta = s_i - s_j
    p_ij = torch.sigmoid(delta)  # Probability that i outranks j

    bce = -(
        labels * torch.log(p_ij + 1e-8)
        + (1 - labels) * torch.log(1 - p_ij + 1e-8)
    )
    return bce.mean()



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

class ENetSimple(nn.Module):
    """
    Simple Evaluation Network (ENetSimple) that takes two contour embeddings
    and predicts the probability that g_i is better than g_j.

    Inputs:
        contour_emb_i: (B, contour_dim)
        contour_emb_j: (B, contour_dim)
    Output:
        prob: (B,)
    """
    def __init__(self,
                 contour_dim=1152,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.1):
        super(ENetSimple, self).__init__()
        
        # Project contour embeddings to hidden dimension
        self.contour_proj = nn.Linear(contour_dim, hidden_dim)
        
        # Define the interaction and classification layers
        # Here, we concatenate the projected embeddings and their element-wise difference
        self.interaction = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            *[
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_layers - 1)
            ]
        )
        
        # Output layer to produce probability
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, img, contour_emb_i, contour_emb_j):
        """
        Forward pass of ENetSimple.

        Args:
            contour_emb_i: Tensor of shape (B, contour_dim)
            contour_emb_j: Tensor of shape (B, contour_dim)

        Returns:
            prob: Tensor of shape (B,) representing P(g_i > g_j)
        """
        # Project contour embeddings
        proj_i = self.contour_proj(contour_emb_i)  # (B, hidden_dim)
        proj_j = self.contour_proj(contour_emb_j)  # (B, hidden_dim)
        
        # Interaction features
        # Common choices: concatenation, element-wise difference, product, etc.
        # Here, we use concatenation and element-wise difference
        combined = torch.cat([
            proj_i,
            proj_j,
            proj_i - proj_j
        ], dim=1)  # (B, 3 * hidden_dim)
        
        # Pass through interaction layers
        hidden = self.interaction(combined)  # (B, hidden_dim)
        
        # Output probability
        prob = self.output_layer(hidden).squeeze(-1)  # (B,)
        
        return prob
    
class ENet(nn.Module):
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
        => prob: (B,)
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
        
        # Concatenate representations
        combined = torch.cat([g_i_rep, g_j_rep], dim=1)  # (B, 2 * hidden_dim)
        
        # Compute probability
        score = self.score_head(combined).squeeze(-1)    # (B,)

        return score,score
    


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
        
        # After processing, we'll concatenate the representations of g_i and g_j
        self.prob_head = nn.Sequential(
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
        => prob: (B,)
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
        
        # Concatenate representations
        combined = torch.cat([g_i_rep, g_j_rep], dim=1)  # (B, 2 * hidden_dim)
        
        # Compute probability
        score = self.score_head(combined).squeeze(-1)    # (B,)
        prob = self.prob_head(combined).squeeze(-1)    # (B,) 
        return score,prob



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

def custom_contrastive_loss(dist_pred, score_i, score_j):
    """
    dist_pred: (P,) predicted distances
    score_i, score_j: (P,) each in {0,1} or partial in (0,1)
    We'll compute an MSE with the 'target distance' from get_target_distance.
    """
    device = dist_pred.device

    # build a list of target distances
    dist_targets = []
    # we must do this on CPU numpy or do it in a vectorized way on GPU
    # for simplicity, we do a loop:
    si_np = score_i.detach().cpu().numpy()
    sj_np = score_j.detach().cpu().numpy()

    for s_i, s_j in zip(si_np, sj_np):
        dist_targets.append(get_target_distance(s_i, s_j))
    
    dist_targets = torch.tensor(dist_targets, device=device, dtype=torch.float32)
    
    # MSE
    loss = F.mse_loss(dist_pred, dist_targets)
    return loss


def sample_pairs_no_diagonal(N, num_pairs, device='cpu'):
    """
    Samples 'num_pairs' valid (i, j) with i != j from range(N).
    Returns tensors (i_idx, j_idx).
    """
    # 1. All indices from 0 to N*N - 1
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
    return i_idx, j_idx

def train_contrastive_epoch(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    lr=1e-4,
    num_pairs=512
):
    """
    Train one epoch using margin-based contrastive loss.
    
    Args:
        model: ContrastiveENet or similar.
        optimizer: torch optimizer (Adam, etc.)
        train_dataset, test_dataset: Iterables of data items
        epoch: current epoch index
        num_pairs: number of random pairs to sample per data item
        margin: the margin for dissimilar pairs
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    total_loss = 0.0
    total_batches = 0

    # ------------- TRAIN -------------
    for data_item in train_dataset:
        # 1) Load data
        # Suppose get_image_and_contours returns:
        #   image_emb: (1,1024,4,4,4) or (1024,4,4,4)
        #   all_g_embs: (N, 1152)
        #   all_scores: (N,) in {0,1} or partial
        image_emb, all_g_embs, all_scores = get_image_and_contours(data_item)
        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)  # => (1,1024,4,4,4)

        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)

        N = all_scores.shape[0]
        if N < 2:
            continue
        
        # 2) Sample random pairs
        pair_indices = torch.randint(low=0, high=N*N, size=(num_pairs,), device=device)
        i_idx = pair_indices // N
        j_idx = pair_indices % N

        # 3) Build "similar vs. dissimilar" labels
        #    Example: label=1 if score_i==score_j, else 0
        score_i = all_scores[i_idx]
        score_j = all_scores[j_idx]


        # 4) Gather contour embeddings
        g_i = all_g_embs[i_idx]
        g_j = all_g_embs[j_idx]

        # repeat image => (P,1024,4,4,4)
        image_batch = image_emb.repeat(g_i.size(0), 1, 1, 1, 1)

        # forward => dist_pred => (P,)
        dist_pred,_ = model(image_batch, g_i, g_j)

        # custom contrastive loss => MSE to the 'desired distance'
        loss_val = custom_contrastive_loss(dist_pred, score_i, score_j)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()
        total_batches += 1

    avg_train_loss = total_loss / max(total_batches, 1)
    print(f"[Epoch {epoch+1}] Train Contrastive Loss = {avg_train_loss:.6f}")

    test_loss = None
    # ------------- TEST / EVAL -------------
    if (epoch+1) % 10 == 0:
        # Evaluate every 5 epochs
        model.eval()
        total_test_loss = 0.0
        test_batches = 0

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
                
                pair_indices_test = torch.randint(low=0, high=N_test*N_test, size=(num_pairs,), device=device)
                i_idx_test = pair_indices_test // N_test
                j_idx_test = pair_indices_test % N_test

                score_i_test = all_scores[i_idx_test]
                score_j_test = all_scores[j_idx_test]

                g_i_test = all_g_embs[i_idx_test]
                g_j_test = all_g_embs[j_idx_test]

                image_batch_test = image_emb.repeat(g_i_test.size(0), 1, 1, 1, 1)
                
                dist_test,_ = model(image_batch_test, g_i_test, g_j_test)
                loss_test_val = custom_contrastive_loss(dist_test, score_i_test, score_j_test)

                total_test_loss  += loss_test_val.item()
                test_batches += 1

            test_loss = total_test_loss  / max(test_batches,1)
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
            print(f"# # # #  Test Contrastive Loss = {test_loss:.6f} # # # #")
    
    return avg_train_loss, test_loss

def train_contrastive_ranking_epoch(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    lr=1e-4,
    num_pairs=512,
    # Hyperparameters to optionally weight ranking loss by pair type:
    alpha = 0.5, 
    lambda_A=0.0,      # (1,1) or (0,0) – no ranking loss
    lambda_B=1.0,      # (1,0) or (0,1)
    lambda_C=1.0,      # (1,d) or (d,1)
    lambda_D=1.0       # (d1,d2)
):
    """
    Train one epoch using a combined contrastive (MSE) loss and an auxiliary ranking loss.
    
    For each data item:
      - Sample random pairs of contours.
      - Compute the predicted distance (dist_pred) via the model.
      - Compute the target distance using get_target_distance.
      - Compute an MSE loss between dist_pred and the target distance.
      - Also compute a binary ranking target (1 if score_i > score_j, else 0)
        and a per-sample ranking loss via binary cross-entropy.
      - The ranking loss is weighted by a per-sample lambda that depends on the pair type:
          * Type A: (1,1) or (0,0) -> λ = lambda_A (usually 0)
          * Type B: (1,0) or (0,1) -> λ = lambda_B
          * Type C: (1,d) or (d,1) -> λ = lambda_C
          * Type D: (d1,d2)         -> λ = lambda_D
      - The final loss is: loss_total = MSE_loss + (weighted average ranking loss).
    
    Returns:
        (avg_train_loss, test_loss) for the epoch.
    """

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    total_loss = 0.0
    total_batches = 0

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
        # pair_indices = torch.randint(low=0, high=N * N, size=(num_pairs,), device=device)
        # i_idx = pair_indices // N
        # j_idx = pair_indices % N
        i_idx, j_idx = sample_pairs_no_diagonal(N, num_pairs, device=device)
        # 3) Get scores for each pair and compute ranking target (1 if score_i > score_j, else 0)
        score_i = all_scores[i_idx]
        score_j = all_scores[j_idx]
        ranking_target = (score_i > score_j).float()  # (P,)

        # 4) Get contour embeddings for each pair
        g_i = all_g_embs[i_idx]
        g_j = all_g_embs[j_idx]

        # Repeat image embedding for each pair: (P,1024,4,4,4)
        image_batch = image_emb.repeat(g_i.size(0), 1, 1, 1, 1)

        # 5) Forward pass: model returns predicted distance (in [0,1])
        dist_pred,prob_pred = model(image_batch, g_i, g_j)  # (P,)

        # 6) Compute target distances (contrastive/regression target)
        dist_targets = []
        si_np = score_i.detach().cpu().numpy()
        sj_np = score_j.detach().cpu().numpy()
        for s_i, s_j in zip(si_np, sj_np):
            dist_targets.append(get_target_distance(s_i, s_j))
        dist_targets = torch.tensor(dist_targets, device=device, dtype=torch.float32)

        mse_loss = F.mse_loss(dist_pred, dist_targets)

        # 7) Build masks for each ranking type:
        # Type A: (1,1) or (0,0)
        mask_A = ((score_i == 0.0) & (score_j == 0.0)) | ((score_i == 1.0) & (score_j == 1.0))
        # Type B: (1,0) or (0,1)
        mask_B = ((score_i == 1.0) & (score_j == 0.0)) | ((score_i == 0.0) & (score_j == 1.0))
        # Type C: one absolute and one partial: (1,d) or (d,1)
        mask_C = ((score_i == 1.0) & (score_j > 0.0) & (score_j < 1.0)) | \
                 ((score_j == 1.0) & (score_i > 0.0) & (score_i < 1.0))
        # Type D: both partial (in (0,1))
        mask_D = ((score_i > 0.0) & (score_i < 1.0) & (score_j > 0.0) & (score_j < 1.0))

        # Build a lambda tensor of shape (P,) with appropriate weight for each pair.
        lambda_tensor = torch.zeros_like(score_i, dtype=torch.float32)
        lambda_tensor[mask_A] = lambda_A
        lambda_tensor[mask_B] = lambda_B
        lambda_tensor[mask_C] = lambda_C
        lambda_tensor[mask_D] = lambda_D

        # Define overall ranking mask as pairs that are not type A (since Type A gets lambda 0)
        mask_ranking = (lambda_tensor > 0)

        # 8) Compute per-sample ranking loss (BCE with reduction='none')
        # Note: We use F.binary_cross_entropy with reduction='none' to get a loss per pair.
        ranking_loss_all = F.binary_cross_entropy_with_logits(prob_pred, ranking_target, reduction='none')
        
        # Compute the weighted average ranking loss only over pairs that are selected.
        if mask_ranking.sum() > 0:            
            ranking_loss = (ranking_loss_all[mask_ranking] * lambda_tensor[mask_ranking]).sum() / score_i.size(0)
        else:
            ranking_loss = 0.0

        # 9) Combine losses
        loss_val = (1-alpha)*mse_loss + alpha * ranking_loss  # (ranking loss now already weighted per sample)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()
        total_batches += 1

    avg_train_loss = total_loss / max(total_batches, 1)
    print(f"[Epoch {epoch+1}] Train Combined Loss = {avg_train_loss:.6f} | MSE  {mse_loss:.6f} | rank  {ranking_loss:.6f}")



    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        test_batches = 0
        ranking_record = 0.0
        mse_record = 0.0

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
                
                # pair_indices_test = torch.randint(low=0, high=N_test * N_test, size=(num_pairs,), device=device)
                # i_idx_test = pair_indices_test // N_test
                # j_idx_test = pair_indices_test % N_test

                i_idx_test, j_idx_test = sample_pairs_no_diagonal(N_test, num_pairs, device=device)
                score_i_test = all_scores[i_idx_test]
                score_j_test = all_scores[j_idx_test]
                ranking_target_test = (score_i_test > score_j_test).float()

                g_i_test = all_g_embs[i_idx_test]
                g_j_test = all_g_embs[j_idx_test]
                image_batch_test = image_emb.repeat(g_i_test.size(0), 1, 1, 1, 1)
                dist_test, prob_test = model(image_batch_test, g_i_test, g_j_test)

                # Compute target distances
                dist_targets_test = []
                si_np_test = score_i_test.detach().cpu().numpy()
                sj_np_test = score_j_test.detach().cpu().numpy()
                for s_i, s_j in zip(si_np_test, sj_np_test):
                    dist_targets_test.append(get_target_distance(s_i, s_j))
                dist_targets_test = torch.tensor(dist_targets_test, device=device, dtype=torch.float32)
                mse_loss_test = F.mse_loss(dist_test, dist_targets_test)

                # Build masks for test pairs
                mask_A_test = ((score_i_test == 0.0) & (score_j_test == 0.0)) | ((score_i_test == 1.0) & (score_j_test == 1.0))
                mask_B_test = ((score_i_test == 1.0) & (score_j_test == 0.0)) | ((score_i_test == 0.0) & (score_j_test == 1.0))
                mask_C_test = ((score_i_test == 1.0) & (score_j_test > 0.0) & (score_j_test < 1.0)) | \
                              ((score_j_test == 1.0) & (score_i_test > 0.0) & (score_i_test < 1.0))
                mask_D_test = ((score_i_test > 0.0) & (score_i_test < 1.0) & (score_j_test > 0.0) & (score_j_test < 1.0))
                lambda_tensor_test = torch.zeros_like(score_i_test, dtype=torch.float32)
                lambda_tensor_test[mask_A_test] = lambda_A
                lambda_tensor_test[mask_B_test] = lambda_B
                lambda_tensor_test[mask_C_test] = lambda_C
                lambda_tensor_test[mask_D_test] = lambda_D
                mask_ranking_test = (lambda_tensor_test > 0)

                ranking_loss_all_test = F.binary_cross_entropy_with_logits(prob_test, ranking_target_test, reduction='none')
                if mask_ranking_test.sum() > 0:
                    ranking_loss_test = (ranking_loss_all_test[mask_ranking_test] * lambda_tensor_test[mask_ranking_test]).sum() /  score_i_test.size(0)
                else:
                    ranking_loss_test = 0.0

                loss_test_val = (1-alpha)*mse_loss_test + alpha * ranking_loss_test
                
                total_test_loss += loss_test_val.item()
                ranking_record  += ranking_loss_test.item()
                mse_record  += mse_loss_test.item()
                test_batches += 1

            test_loss = total_test_loss / max(test_batches, 1)
            rank_avg_loss = ranking_record / max(test_batches, 1)
            mse_avg_loss = mse_record / max(test_batches, 1)

            end_time = time.time()
            elapsed = end_time - start_time
            print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
            print(f"# # # # Test Combined Loss = {test_loss:.6f} # # # #")
            print(f"# # # # Current test mse {mse_avg_loss:.6f} |  ranking {rank_avg_loss:.6f} # # # #")
    return avg_train_loss, test_loss



def train_neighbourwise_epoch(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    lr=1e-4,
    skip_equal = False
):
    """
    Trains `ENet` or `ENetSimple` for one epoch using neighbor-wise pairs from the dataset.
    If `loop=True`, then we do per-pair forward/backward in a loop (like train_neighbourwise_epoch_loop).
    If `loop=False`, we do a single batched forward/backward for all pairs (like train_neighbourwise_epoch).
    Evaluation is done every 10 epochs.

    Args:
        model: either `ENet` (requires image_emb) or `ENetSimple` (ignores image_emb).
        optimizer: torch optimizer (e.g., Adam).
        train_dataset, test_dataset: Iterable data sources.
        epoch: current epoch number (0-based).
        lr: learning rate (optional, typically set in optimizer init).
        loop: if True, do a loop-based approach (pair by pair), otherwise a batched approach.
    Returns:
        (avg_train_loss, test_loss_if_any)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    start_time = time.time()
    epoch_loss = 0.0
    num_pairs = 0  # We'll track how many neighbor pairs were processed

    # --- Training ---
    for data_item in train_dataset:
        # 1) Load data
        #    get_image_and_contours_neighbour is presumably a function that returns:
        #    image_emb: shape (1, 1024,4,4,4) or (1024,4,4,4)
        #    all_g_embs: shape (N, 1152)
        #    all_scores: shape (N,)
        image_emb, all_g_embs, all_scores = get_image_and_contours_neighbour(data_item)
        N = all_scores.shape[0]
        if N < 2:
            continue

        if len(image_emb.shape) == 4:
            # make sure image_emb has batch=1
            image_emb = image_emb.unsqueeze(0)  # => (1,1024,4,4,4)

        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)

        # 2) Neighbor pairs: (0,1), (1,2), ..., (N-2, N-1)
        pair_perm = torch.randperm(N - 1, device=device)
        pre_i_idx = pair_perm
        pre_j_idx = pair_perm + 1  # be sure not to exceed N-1

        # 3) Randomly flip each pair with 50% probability
        flip_mask = torch.rand(pre_i_idx.size(0), device=device) < 0.5  # (N-1,)
        i_idx = torch.where(flip_mask, pre_i_idx, pre_j_idx)  # If flip_mask=True, keep i; else j
        j_idx = torch.where(flip_mask, pre_j_idx, pre_i_idx)  # If flip_mask=True, keep j; else i


        if skip_equal:
            # 3) Skip pairs where the scores are equal
            score_i = all_scores[i_idx]
            score_j = all_scores[j_idx]
            valid_mask =  (score_i != score_j)
            
            i_idx = i_idx[valid_mask]
            j_idx = j_idx[valid_mask]
        if i_idx.size(0) == 0:
            continue

        # -- Batched approach (all neighbor pairs in one pass) --
        g_i = all_g_embs[i_idx]  # (N-1,1152)
        g_j = all_g_embs[j_idx]  # (N-1,1152)
        labels = (all_scores[i_idx] > all_scores[j_idx]).float()  # (N-1,)

        # Expand image batch to match (N-1) pairs
        image_batch = image_emb.repeat(g_i.size(0), 1, 1, 1, 1)  # => (N-1,1024,4,4,4)
        pred_probs,_ = model(image_batch, g_i, g_j)  # => (N-1,)

        loss_val = F.binary_cross_entropy_with_logits(pred_probs, labels)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()


        epoch_loss += loss_val.item()
        # number of pairs processed
        num_pairs += 1

    avg_train_loss = epoch_loss / max(num_pairs, 1)
    print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.6f}")

    # --- Evaluation every 10 epochs ---
    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        num_test_items = 0

        with torch.no_grad():
            for val_data_item in test_dataset:
                image_emb, all_g_embs, all_scores = get_image_and_contours_neighbour(val_data_item)
                N_test = all_scores.shape[0]
                if N_test < 2:
                    continue

                if len(image_emb.shape) == 4:
                    image_emb = image_emb.unsqueeze(0)
                image_emb = image_emb.to(device)

                all_g_embs = all_g_embs.to(device)
                all_scores = all_scores.to(device)

                # neighbor pairs
                i_idx_test = torch.arange(N_test - 1, device=device)
                j_idx_test = i_idx_test + 1
                if skip_equal:
                    # 3) Skip pairs where the scores are equal
                    score_i = all_scores[i_idx_test]
                    score_j = all_scores[j_idx_test]
                    valid_mask_test =  (score_i != score_j)
                    
                    i_idx_test = i_idx_test[valid_mask_test]
                    j_idx_test = j_idx_test[valid_mask_test]
                if i_idx_test.size(0) == 0:
                    continue
                g_i_test = all_g_embs[i_idx_test]
                g_j_test = all_g_embs[j_idx_test]

                labels_test = (all_scores[i_idx_test] > all_scores[j_idx_test]).float()

         
                image_batch_test = image_emb.repeat(g_i_test.size(0), 1, 1, 1, 1)
                pred_test_probs,_ = model(image_batch_test, g_i_test, g_j_test)
    
                loss_test_val = F.binary_cross_entropy_with_logits(pred_test_probs, labels_test)
                total_test_loss += loss_test_val.item()
                num_test_items += 1

        test_loss = total_test_loss / max(num_test_items, 1)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
        print(f"# # # # Test Loss = {test_loss:.6f} # # # #")

    return avg_train_loss, test_loss



def train_randompairs_epoch(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    num_pairs=512,
    lr=1e-4,
    loop=False,
    skip_equal = True
):
    """
    Trains `ENet` or `ENetSimple` for one epoch using random pairwise comparisons from the dataset.
    If `loop=True`, we do a per-pair forward/backward (as in train_randompairs_epoch_loop).
    If `loop=False`, we do a single batched forward/backward (as in train_randompairs_epoch).
    Evaluates every 10 epochs using the same random-pair approach.

    Args:
        model: ENet or ENetSimple instance.
        optimizer: torch optimizer (e.g., Adam).
        train_dataset, test_dataset: Iterable of data items.
        epoch: current epoch index (0-based).
        num_pairs: number of random pairs to sample per data_item (default 512).
        lr: learning rate (set in optimizer or here).
        loop: if True, do a loop-based approach. If False, a batched approach.
    Returns:
        (avg_train_loss, test_loss_if_any)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    start_time = time.time()
    total_loss = 0.0
    total_pairs = 0

    # --- Training ---
    for data_item in train_dataset:
        # 1) Load data
        #    get_image_and_contours is presumably a function that returns:
        #    image_emb: shape (1,1024,4,4,4) or (1024,4,4,4)
        #    all_g_embs: shape (N, 1152)
        #    all_scores: shape (N,)
        image_emb, all_g_embs, all_scores = get_image_and_contours(data_item)
        N = all_scores.shape[0]
        if N < 2:
            continue

        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)  # => (1,1024,4,4,4)

        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)

        # 2) Sample random pairs
        pair_indices = torch.randint(low=0, high=N*N, size=(num_pairs,), device=device)
        i_idx = pair_indices // N
        j_idx = pair_indices % N

        # 3) Skip pairs where the scores are equal

        if skip_equal:
            # 3) Skip pairs where the scores are equal
            score_i = all_scores[i_idx]
            score_j = all_scores[j_idx]
            valid_mask = (score_i != score_j)
            
            i_idx = i_idx[valid_mask]
            j_idx = j_idx[valid_mask]
        if i_idx.size(0) == 0:
            continue


        # -- Batched approach --
        if i_idx.numel() == 0:
            continue  # no valid pairs
        g_i = all_g_embs[i_idx]  # shape => (P,1152)
        g_j = all_g_embs[j_idx]  # shape => (P,1152)
        labels = (all_scores[i_idx] > all_scores[j_idx]).float()  # (P,)

        image_batch = image_emb.repeat(labels.size(0), 1, 1, 1, 1)  # => (P,1024,4,4,4)
        pred_probs,_ = model(image_batch, g_i, g_j)  # => (P,)

        loss_val = F.binary_cross_entropy_with_logits(pred_probs, labels)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()
        total_pairs += 1

    avg_train_loss = total_loss / max(total_pairs, 1)
    print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.6f}")
    # --- Evaluate every 10 epochs ---
    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        num_test_items = 0

        with torch.no_grad():
            for val_data_item in test_dataset:
                image_emb, all_g_embs, all_scores = get_image_and_contours(val_data_item)
                N_test = all_scores.shape[0]
                if N_test < 2:
                    continue

                if len(image_emb.shape) == 4:
                    image_emb = image_emb.unsqueeze(0)
                image_emb = image_emb.to(device)

                all_g_embs = all_g_embs.to(device)
                all_scores = all_scores.to(device)

                # random pairs
                pair_indices_test = torch.randint(
                    0, N_test*N_test, (num_pairs,), device=device
                )
                i_idx_test = pair_indices_test // N_test
                j_idx_test = pair_indices_test % N_test


                if skip_equal:
                    # 3) Skip pairs where the scores are equal
                    score_i = all_scores[i_idx_test]
                    score_j = all_scores[j_idx_test]
                    valid_mask_test =  (score_i != score_j)
                    
                    i_idx_test = i_idx_test[valid_mask_test]
                    j_idx_test = j_idx_test[valid_mask_test]

                if i_idx_test.size(0) == 0:
                    continue

                g_i_test = all_g_embs[i_idx_test]
                g_j_test = all_g_embs[j_idx_test]
                labels_test = (all_scores[i_idx_test] > all_scores[j_idx_test]).float()

   
                image_batch_test = image_emb.repeat(labels_test.size(0), 1, 1, 1, 1)
                pred_test_probs,_ = model(image_batch_test, g_i_test, g_j_test)
            

                loss_test_val = F.binary_cross_entropy_with_logits(pred_test_probs, labels_test)
                total_test_loss += loss_test_val.item()
                num_test_items += 1

        test_loss = total_test_loss / max(num_test_items, 1)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
        print(f"# # # # Test Loss = {test_loss:.6f} # # # #")

    return avg_train_loss, test_loss


