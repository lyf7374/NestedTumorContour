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
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
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
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
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
        prob = self.score_head(combined).squeeze(-1)    # (B,)
        return prob
    
def train_neighbourwise_epoch(
    model,
    optimizer,
    train_dataset,
    test_dataset,
    epoch,
    lr=1e-4
):
    """
    Trains `ENet` or `ENetSimple` for one epoch using neighbor-wise pairs from exactly 150 contours.

    For each data_item in train_dataset:
      1. Load image_emb (1,1024,4,4,4) and 150 contour_embs (150,1152), plus scores (150,).
      2. Generate consecutive pairs (i, i+1) => total 149 pairs.
      3. Model forward => probability p_{i,i+1} for each pair.
      4. Compare p_{i,i+1} to the label (scores[i] > scores[i+1] => 1 else 0).
      5. Accumulate BCE loss, backprop, optimizer step.

    Evaluation is done every 10 epochs, using the same neighbor-wise approach on test_dataset.

    Args:
        model: either `ENet` (requires image_emb) or `ENetSimple` (ignores image_emb).
        optimizer: torch optimizer (e.g., Adam).
        train_dataset, test_dataset: Iterable data sources.
        epoch: current epoch number (0-based).
        lr: learning rate (optional, typically set in optimizer init).
        device: torch device (CPU/GPU). Autodetect if None.

    Returns:
        (avg_train_loss, test_loss_if_any)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    start_time = time.time()
    epoch_loss = 0.0
    num_items = 0

    # --- Training ---
    for data_item in train_dataset:
        # 1) Load data
        image_emb, all_g_embs, all_scores = get_image_and_contours_neighbour(data_item)
        # image_emb : (1024,4,4,4)  or sometimes (1,1024,4,4,4) => ensure shape
        # all_g_embs: (150, 1152)
        # all_scores: (150,)

        N = all_scores.shape[0]
        if N < 2:
            continue

        # Move to device
        # If using ENet (with image), we keep shape (B=1, 1024,4,4,4):
        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)  # => (1,1024,4,4,4)
        image_emb = image_emb.to(device)

        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)

        # 2) Create neighbor pairs (0,1), (1,2), ..., (N-2, N-1)
        i_idx = torch.arange(N-1, device=device)
        j_idx = i_idx + 1  # shape (N-1,)

        # 3) Gather embeddings for these pairs
        # shape => (N-1, 1152)
        g_i = all_g_embs[i_idx]
        g_j = all_g_embs[j_idx]

        # 4) Construct binary labels: 1 if scores[i] > scores[j], else 0
        labels = (all_scores[i_idx] > all_scores[j_idx]).float()  # shape (N-1,)

        # 5) Forward pass
  
        image_batch = image_emb.repeat(g_i.size(0), 1, 1, 1, 1)  # => (N-1, 1024,4,4,4)
        pred_probs = model(image_batch, g_i, g_j)  # => (N-1,)
 

        # 6) Compute BCE loss
        loss_val = F.binary_cross_entropy(pred_probs, labels)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        epoch_loss += loss_val.item()
        num_items += 1

    avg_train_loss = epoch_loss / max(num_items, 1)
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

                g_i_test = all_g_embs[i_idx_test]
                g_j_test = all_g_embs[j_idx_test]

                labels_test = (all_scores[i_idx_test] > all_scores[j_idx_test]).float()

         
                image_batch_test = image_emb.repeat(g_i_test.size(0), 1, 1, 1, 1)
                pred_test_probs = model(image_batch_test, g_i_test, g_j_test)
    
                loss_test_val = F.binary_cross_entropy(pred_test_probs, labels_test)
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
    lr=1e-4
):
    """
    Trains `ENet` or `ENetSimple` for one epoch using random pairwise comparisons from 800 contours.

    For each data_item in train_dataset:
      1. Load image_emb (1,1024,4,4,4) and 800 contour_embs (800,1152), plus scores (800,).
      2. Sample random pairs (i_idx, j_idx).
      3. Model forward => probability p_{i,j}.
      4. Compare p_{i,j} to label (scores[i]>scores[j]?).
      5. BCE loss, backprop, optimizer step.

    Evaluates every 10 epochs similarly with random pairs on the test set.

    Args:
        model: either `ENet` (needs image_emb) or `ENetSimple` (ignores image_emb).
        optimizer: torch optimizer.
        train_dataset, test_dataset: Iterable data sources.
        epoch: current epoch number (0-based).
        num_pairs: how many pairs to sample per data_item.
        lr: learning rate.
        device: torch device (CPU/GPU). Autodetect if None.

    Returns:
        (avg_train_loss, test_loss_if_any)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    start_time = time.time()
    epoch_loss = 0.0
    num_items = 0

    # --- Training ---
    for data_item in train_dataset:
        # 1) Load data
        image_emb, all_g_embs, all_scores = get_image_and_contours(data_item)
        N = all_scores.shape[0]  # Expect 800 typically

        if N < 2:
            continue

        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)  # => (1,1024,4,4,4)
        image_emb = image_emb.to(device)

        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)

        # 2) Randomly sample `num_pairs` pairs
        # Each pair_indices[k] is in [0, N*N)
        pair_indices = torch.randint(
            low=0, high=N*N, size=(num_pairs,), device=device
        )
        i_idx = pair_indices // N
        j_idx = pair_indices % N

        # optionally ensure i_idx != j_idx
        valid_mask = (i_idx != j_idx)
        i_idx = i_idx[valid_mask]
        j_idx = j_idx[valid_mask]

        # gather embeddings
        g_i = all_g_embs[i_idx]
        g_j = all_g_embs[j_idx]

        # gather labels
        labels = (all_scores[i_idx] > all_scores[j_idx]).float()

        if labels.shape[0] == 0:
            continue

        # 3) Forward pass
 
        image_batch = image_emb.repeat(labels.size(0), 1, 1, 1, 1)
        pred_probs = model(image_batch, g_i, g_j)  # => (P,)


        # 4) Compute BCE
        loss_val = F.binary_cross_entropy(pred_probs, labels)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        epoch_loss += loss_val.item()
        num_items += 1

    avg_train_loss = epoch_loss / max(num_items, 1)
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

                valid_mask_test = (i_idx_test != j_idx_test)
                i_idx_test = i_idx_test[valid_mask_test]
                j_idx_test = j_idx_test[valid_mask_test]

                if i_idx_test.size(0) == 0:
                    continue

                g_i_test = all_g_embs[i_idx_test]
                g_j_test = all_g_embs[j_idx_test]
                labels_test = (all_scores[i_idx_test] > all_scores[j_idx_test]).float()

   
                image_batch_test = image_emb.repeat(labels_test.size(0), 1, 1, 1, 1)
                pred_test_probs = model(image_batch_test, g_i_test, g_j_test)
            

                loss_test_val = F.binary_cross_entropy(pred_test_probs, labels_test)
                total_test_loss += loss_test_val.item()
                num_test_items += 1

        test_loss = total_test_loss / max(num_test_items, 1)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"# # # # Evaluation for Epoch {epoch+1} completed in {elapsed:.2f}s # # # #")
        print(f"# # # # Test Loss = {test_loss:.6f} # # # #")

    return avg_train_loss, test_loss
