import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

bce_loss_fn = nn.BCEWithLogitsLoss()  
def model_load(model, dic_pth):
    state_dict = torch.load(dic_pth, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix if it exists
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

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
# 2) Transformer Decoder Model (ListwiseDecoderModel)
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
        
        # FF
        ff_out = self.ff(contour_tokens)
        contour_tokens = self.norm3(contour_tokens + ff_out)
        return contour_tokens

class ListwiseDecoderModel(nn.Module):
    """
    A "listwise" model with:
      - Projected image tokens => (B,64,hidden_dim)
      - Projected contour tokens => (B,K,hidden_dim)
      - Stacked cross-attn blocks
      - Scores => (B,K)
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
        
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, image_emb, contour_embs):
        """
        image_emb: (B,1024,4,4,4)
        contour_embs: (B,K,1152)
        => scores: (B,K)
        """
        B, C, D, H, W = image_emb.shape
        # Flatten => (B,64,1024)
        img_tokens = image_emb.view(B, C, D*H*W).transpose(1,2)  # => (B,64,1024)
        img_tokens = self.img_proj(img_tokens)                   # => (B,64,hidden_dim)
        
        contour_tokens = self.contour_proj(contour_embs)         # => (B,K,hidden_dim)
        
        # pass through multiple layers
        for layer in self.layers:
            contour_tokens = layer(contour_tokens, img_tokens)
        
        # final score => (B,K,1) => (B,K)
        scores = self.score_head(contour_tokens).squeeze(-1)
        return scores

#############################################################
# 3) Training Loop for Pairwise BCE
#############################################################
def train_pairwise_epoch(model,
                         optimizer,
                         train_dataset,
                         test_dataset,
                         epoch,
                         num_pairs=64,
                         lr=1e-4):
    """
    Each iteration processes 1 item => (800) contours => sample pairs => BCE.
    We evaluate every 10 epochs on test_dataset.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    epoch_loss = 0.0

    # --- Training ---
    for data_item in train_dataset:
        # data_item => dict from HDF5 => get 800 contours
        img_emb, all_g_embs, all_scores = get_image_and_contours(data_item)
        N = all_scores.shape[0]  # 800

        # Move to device
        img_emb = img_emb.unsqueeze(0).to(device)       # => (1,1024,4,4,4)
        all_g_embs = all_g_embs.unsqueeze(0).to(device) # => (1,800,1152)
        all_scores = all_scores.to(device)              # => (800,)

        # Random pairs
        pair_indices = torch.randint(low=0, high=N*N, size=(num_pairs,), device=device)
        i_idx = pair_indices // N
        j_idx = pair_indices % N
        
        # Forward => shape (1,800) => squeeze => (800,)
        pred_scores = model(img_emb, all_g_embs).squeeze(0)

        # Pairwise BCE
        loss_val = pairwise_bce_loss(pred_scores, all_scores, i_idx, j_idx)

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        epoch_loss += loss_val.item()

    avg_train_loss = epoch_loss / max(len(train_dataset), 1)
    print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.6f}")

    # --- Evaluate every 10 epochs ---
    test_loss = None
    if (epoch+1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for val_data_item in test_dataset:
                img_emb, all_g_embs, all_scores = get_image_and_contours(val_data_item)
                N_test = all_scores.shape[0]
                img_emb = img_emb.unsqueeze(0).to(device)
                all_g_embs = all_g_embs.unsqueeze(0).to(device)
                all_scores = all_scores.to(device)
                
                # pair_indices_test = torch.randint(0, N_test*N_test, (num_pairs,), device=device)
                pair_indices_test = torch.randint(0, N_test*N_test, (800,), device=device)
                i_idx_test = pair_indices_test // N_test
                j_idx_test = pair_indices_test % N_test

                pred_test_scores = model(img_emb, all_g_embs).squeeze(0)
                loss_test_val = pairwise_bce_loss(pred_test_scores, all_scores,
                                                  i_idx_test, j_idx_test)
                total_test_loss += loss_test_val.item()
        test_loss = total_test_loss / max(len(test_dataset), 1)
        end_time = time.time()
        eval_elapsed = end_time - start_time
        print(f"# # # # # # Evaluation for Epoch {epoch+1} completed in {eval_elapsed:.2f} seconds.# # # # # # ")
        print(f"# # # # Test Loss = {test_loss:.6f} # # # # # ")

    return avg_train_loss, test_loss

def lambda_rank_loss(pred_scores, true_scores, alpha=1.0, epsilon=1e-10):
    """
    A revised 2D-based LambdaRank-like loss that maintains (K,K) shapes.
    
    pred_scores: (K,) predicted
    true_scores: (K,) ground-truth in [0,1]
    """
    device = pred_scores.device
    K = pred_scores.size(0)
    
    # Sort predicted scores => to get predicted rank ordering
    # (for delta_ndcg calculation)
    _, sorted_indices = torch.sort(pred_scores, descending=True)
    
    # ranks is the "predicted rank" of each item
    # We'll create a rank array of shape (K,)
    # sorted_indices[i] = index of item that is i-th in sorted order
    # We want ranks[index_of_item] = i
    ranks = torch.empty_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.arange(K, device=device).float() + 1.0  # 1-based
    
    # 1) Build 2D differences
    pred_diff = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(0)  # (K,K)
    true_diff = true_scores.unsqueeze(1) - true_scores.unsqueeze(0)  # (K,K)
    
    # 2) Sigmoid factor => shape (K,K)
    # sign is reversed so that if pred_scores[i] < pred_scores[j] but i > j in ground truth, we push them
    rho_2d = torch.sigmoid(-alpha * pred_diff)
    
    # 3) delta_ndcg => shape (K,K)
    # We'll compute rank_i, rank_j as the predicted rank for each item i,j
    # rank array has shape (K,), so we do broadcasting
    rank_i = ranks.unsqueeze(1).expand(K, K)  # (K,K)
    rank_j = ranks.unsqueeze(0).expand(K, K)  # (K,K)
    delta_ndcg_2d = torch.abs(
        (1.0 / torch.log2(rank_i + 1.0 + epsilon)) -
        (1.0 / torch.log2(rank_j + 1.0 + epsilon))
    )
    
    # 4) We only care about pairs with true_scores[i] > true_scores[j]
    mask_2d = (true_diff > 0).float()  # (K,K), 1 where i>j, else 0
    
    # Combine => shape (K,K)
    comb_2d = delta_ndcg_2d * rho_2d * mask_2d
    
    # 5) sum over j => shape (K,)
    lambda_i = comb_2d.sum(dim=1)  # row-wise sum => each i's total gradient
    
    # 6) Define a loss. 
    # Commonly, in “lambda” approaches, we do: 
    # L = (lambda_i * pred_scores_i).sum() or 
    # L = - (lambda_i * pred_scores_i).sum() 
    # depending on your sign convention.
    loss = (lambda_i * pred_scores).sum()
    
    return loss


def train_lambdarank_epoch(model,
                           optimizer,
                           train_dataset,
                           test_dataset,
                           epoch,
                           lr=1e-4):
    """
    Trains a cross-attention model with a LambdaRank-like (listwise) loss.

    The structure is analogous to the pairwise BCE approach: we process
    each data item from 'train_dataset' in turn (one epoch), compute a 
    ranking loss, and do a forward/backward pass. 
    Every 10 epochs, we evaluate on 'test_dataset'.
    
    Args:
        model (nn.Module): 
            The cross-attention ranking model, e.g. CrossAttnListwiseModel.
        optimizer (torch.optim.Optimizer):
            The optimizer for backprop updates.
        train_dataset (Iterable):
            Each item is a dict with image embedding and contour embeddings.
        test_dataset (Iterable):
            For evaluation every 10 epochs.
        epoch (int):
            Current epoch index (0-based or 1-based).
        lr (float):
            Learning rate if needed, not strictly used here.

    Returns:
        (float, float):
            (avg_train_loss, test_loss). If no test is done this epoch,
            test_loss=None.
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    epoch_loss = 0.0

    # --- Training over the entire train_dataset ---
    for data_item in train_dataset:
        # 1) Load image & up to N=800 contours
        img_emb, all_g_embs, all_scores = get_image_and_contours(data_item)
        N = all_scores.shape[0]  # typically 800

        # 2) Move to device
        #    For a single data item, we unsqueeze(0) to create a batch dimension => (1,1024,4,4,4)
        img_emb = img_emb.unsqueeze(0).to(device)        # => (1,1024,4,4,4)
        all_g_embs = all_g_embs.unsqueeze(0).to(device)  # => (1,N,1152)
        all_scores = all_scores.to(device)               # => (N,)

        # 3) Forward => shape (1,N) => squeeze => shape (N,)
        pred_scores = model(img_emb, all_g_embs).squeeze(0)  # => (N,)

        # 4) Compute LambdaRank-like loss
        loss_val = lambda_rank_loss(pred_scores, all_scores)

        # 5) Backprop
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # 6) Accumulate
        epoch_loss += loss_val.item()

    # Average training loss
    avg_train_loss = epoch_loss / max(len(train_dataset), 1)
    print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.6f}")


    # --- Evaluate every 10 epochs ---
    test_loss = None
    if (epoch+1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for val_data_item in test_dataset:
                img_emb, all_g_embs, all_scores = get_image_and_contours(val_data_item)
                N_test = all_scores.shape[0]
                img_emb = img_emb.unsqueeze(0).to(device)
                all_g_embs = all_g_embs.unsqueeze(0).to(device)
                all_scores = all_scores.to(device)
                
                pair_indices_test = torch.randint(0, N_test*N_test, (800,), device=device)
                i_idx_test = pair_indices_test // N_test
                j_idx_test = pair_indices_test % N_test

                pred_test_scores = model(img_emb, all_g_embs).squeeze(0)
                loss_test_val = pairwise_bce_loss(pred_test_scores, all_scores,
                                                  i_idx_test, j_idx_test)
                total_test_loss += loss_test_val.item()
        test_loss = total_test_loss / max(len(test_dataset), 1)
        end_time = time.time()
        eval_elapsed = end_time - start_time
        print(f"# # # # # # Evaluation for Epoch {epoch+1} completed in {eval_elapsed:.2f} seconds.# # # # # # ")
        print(f"# # # # Test Loss = {test_loss:.6f} # # # # # ")


    # # --- Evaluate on test_dataset every 10 epochs ---
    # test_loss = None
    # if (epoch + 1) % 10 == 0:
    #     model.eval()
    #     total_test_loss = 0.0
    #     with torch.no_grad():
    #         for val_data_item in test_dataset:
    #             img_emb, all_g_embs, all_scores = get_image_and_contours(val_data_item)
    #             N_test = all_scores.shape[0]

    #             # Same shape logic => (1, 1024,4,4,4), (1,N_test,1152), (N_test,)
    #             img_emb = img_emb.unsqueeze(0).to(device)
    #             all_g_embs = all_g_embs.unsqueeze(0).to(device)
    #             all_scores = all_scores.to(device)

    #             # Predict => (N_test,)
    #             pred_test_scores = model(img_emb, all_g_embs).squeeze(0)
                
    #             # Compute LambdaRank-like test loss
    #             loss_test_val = lambda_rank_loss(pred_test_scores, all_scores)
    #             total_test_loss += loss_test_val.item()

    #     test_loss = total_test_loss / max(len(test_dataset), 1)
    #     end_time = time.time()
    #     eval_elapsed = end_time - start_time
    #     print(f"# # # # # # Evaluation for Epoch {epoch+1} completed in {eval_elapsed:.2f} seconds.# # # # # # ")
    #     print(f"# # # # Test Loss = {test_loss:.6f} # # # # # ")

    return avg_train_loss, test_loss


  
