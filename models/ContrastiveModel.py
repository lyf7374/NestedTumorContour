import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

bce_loss_fn = nn.BCEWithLogitsLoss()  


class PairwiseContrastiveModel(nn.Module):
    def __init__(self, img_dim=1024, contour_dim=1152, hidden_dim=256):
        super().__init__()
        
        # 1) Image projection: we'll do a linear to reduce 1024 -> 256
        self.img_pool = nn.AdaptiveAvgPool3d((1,1,1))  # to handle (C,D,H,W) -> (C,1,1,1)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 2) Contour projection: 1152 -> 256
        self.contour_proj = nn.Sequential(
            nn.Linear(contour_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 3) Fusion + final scoring
        # We'll just concat the two 256-d vectors => 512 => 1
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # final scalar
        )
        
    def forward(self, img_emb, contour_emb):
        """
        img_emb:    (1024, 4, 4, 4)  or (B, 1024, 4, 4)
        contour_emb:(N, 1152)        or (B, N, 1152)
        
        We'll produce a scalar for each contour in the batch.
        """
        # 1) Pool the image embedding => shape (1024,) or (B,1024)
        if img_emb.dim() == 4:
            # shape is (C,D,H,W), add a batch dimension
            img_emb = img_emb.unsqueeze(0)  # => (1, 1024, 4, 4, 4)
        # If it has shape (B, C, D, H, W), that's also fine.
        
        # We'll handle both single-image and mini-batch cases:
        B = img_emb.shape[0]  # either 1 or your batch size
        # Pool: (B, 1024, 1,1,1) => (B, 1024)
        img_vec = self.img_pool(img_emb).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 1024)
        
        # 2) Project image => (B, hidden_dim)
        img_vec = self.img_proj(img_vec)
        
        # 3) If we have multiple contours, we want to produce a score for each
        # Let’s assume shape (N, 1152) for a single image, or (B, N, 1152).
        if contour_emb.dim() == 2:
            # single image, multiple contours
            N = contour_emb.shape[0]
            c_vec = self.contour_proj(contour_emb)  # (N, hidden_dim)
            # We'll "broadcast" img_vec over all N:
            #   img_vec is (1, hidden_dim), c_vec is (N, hidden_dim)
            img_vec = img_vec.repeat(N, 1)  # => (N, hidden_dim)
            
        elif contour_emb.dim() == 3:
            # (B, N, 1152)
            B, N, _ = contour_emb.shape
            # flatten => (B*N, 1152)
            c_vec = contour_emb.view(B*N, -1)
            c_vec = self.contour_proj(c_vec)  # => (B*N, hidden_dim)
            # expand img_vec => (B*N, hidden_dim)
            img_vec = img_vec.unsqueeze(1).repeat(1, N, 1).view(B*N, -1)
        else:
            raise ValueError("contour_emb must be 2D or 3D.")
        
        # 4) Fusion => final score
        fusion_inp = torch.cat([img_vec, c_vec], dim=-1)  # (N, 512) or (B*N, 512)
        score = self.fusion(fusion_inp)  # => (N,1) or (B*N,1)
        return score.squeeze(-1)  # shape (N,) or (B*N,)

    def score_single(self, img_emb, single_contour_emb):
        """ Utility: produce a single scalar for one contour. """
        return self.forward(img_emb, single_contour_emb.unsqueeze(0))[0]
    

class CrossAttnPairwiseModel(nn.Module):
    def __init__(self, 
                 img_channels=1024, 
                 num_img_tokens=64,  # 4*4*4 = 64
                 contour_dim=1152,
                 hidden_dim=256,
                 n_heads=4):
        super().__init__()
        
        self.img_channels = img_channels
        self.num_img_tokens = num_img_tokens
        self.hidden_dim = hidden_dim
        
        # 1) Linear projection for image tokens (1024 -> hidden_dim)
        self.img_token_proj = nn.Linear(img_channels, hidden_dim)
        
        # 2) Linear projection for contour embedding (1152 -> hidden_dim)
        self.contour_proj = nn.Linear(contour_dim, hidden_dim)
        
        # 3) Multi-head attention module
        # We'll treat each contour as a single query of shape (1, hidden_dim).
        # The image is the key/value of shape (64, hidden_dim).
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                num_heads=n_heads,
                                                batch_first=True)  
        # `batch_first=True` => input shape (B, seq_len, embed_dim).
        
        # 4) Optional feed-forward or MLP after cross attention
        self.post_attn_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 5) Final scoring head => scalar
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, img_emb, contour_embs):
        """
        img_emb:      shape (1024, 4, 4, 4) or (1, 1024, 4, 4, 4)
                      We'll flatten to (1, 64, 1024) or (B,64,1024).
        
        contour_embs: shape (N, 1152) or (B, N, 1152)
                      We'll produce a scalar for each contour in the batch.
        """
        # Ensure batch dimension:
        if img_emb.dim() == 4:  
            # shape (C,D,H,W) => treat as single-batch
            img_emb = img_emb.unsqueeze(0)  # (1,1024,4,4,4)

        B = img_emb.shape[0]  # batch size for images

        # 1) Flatten image embedding => (B, 64, 1024)
        #    Suppose dimension ordering is (B, C, D, H, W). We'll flatten (D*H*W)=64
        #    to get shape (B, 64, C). A generic flatten:
        img_tokens = img_emb.view(B, self.img_channels, -1).transpose(1,2)
        # now shape (B, 64, 1024)
        
        # 2) Project image tokens => (B, 64, hidden_dim)
        img_tokens_proj = self.img_token_proj(img_tokens)
        
        # 3) Handle contour embeddings
        #    If shape is (N,1152), we'll treat B=1. If shape is (B,N,1152), we do batch.
        if contour_embs.dim() == 2:
            # (N,1152) => (1,N,1152)
            contour_embs = contour_embs.unsqueeze(0)  # (1,N,1152)
        
        # contour_embs now shape (B, N, 1152)
        B2, N, _ = contour_embs.shape
        assert B2 == B, "Image batch size and contour batch size must match"

        # Flatten to treat each contour as a separate query in a big batch
        # shape => (B*N, 1152)
        contour_embs_flat = contour_embs.view(B*N, -1)
        
        # Project to hidden_dim => (B*N, hidden_dim)
        contour_queries = self.contour_proj(contour_embs_flat)
        
        # We'll pass each query separately to cross-attention. But that
        # can be done in a single call if we replicate the image tokens for each query:
        #    - Expand img_tokens_proj from (B, 64, hidden_dim)
        #      to (B*N, 64, hidden_dim)
        img_tokens_expanded = img_tokens_proj.unsqueeze(1)  # (B,1,64,hidden_dim)
        img_tokens_expanded = img_tokens_expanded.repeat(1, N, 1, 1)  # (B,N,64,hidden_dim)
        # flatten => (B*N,64,hidden_dim)
        img_tokens_expanded = img_tokens_expanded.view(B*N, self.num_img_tokens, self.hidden_dim)
        
        # Each query we have shape (B*N, 1, hidden_dim).
        # We need to reshape contour_queries => (B*N, 1, hidden_dim) for attention
        contour_queries = contour_queries.unsqueeze(1)  # (B*N,1,hidden_dim)
        
        # 4) Cross-attention
        #    Q = contour_queries, K=V = expanded image tokens.
        #    multiheadAttention expects shape (batch, seq_len, embed_dim).
        #    We'll pass the query as (B*N, 1, hidden_dim),
        #    keys/values as (B*N, 64, hidden_dim).
        #    The output will be (B*N, 1, hidden_dim).
        attn_output, _ = self.cross_attn(query=contour_queries, 
                                         key=img_tokens_expanded, 
                                         value=img_tokens_expanded)
        # shape of attn_output = (B*N, 1, hidden_dim)
        
        # Remove seq dimension => (B*N, hidden_dim)
        fused_rep = attn_output.squeeze(1)  # (B*N, hidden_dim)
        
        # Optional feed-forward
        fused_rep = self.post_attn_ff(fused_rep)  # (B*N, hidden_dim)
        
        # 5) Final scoring => (B*N,1)
        scores = self.score_head(fused_rep).squeeze(-1)  # shape (B*N,)
        
        # Reshape back to (B, N) if you want
        scores = scores.view(B, N)  # (B,N)
        return scores  # shape (B,N)



def pairwise_bce_loss(pred_scores, true_scores, i_idx, j_idx):
    """
    pred_scores: shape (N,) => predicted score for each contour
    true_scores: shape (N,) => ground-truth in [0,1]
    i_idx, j_idx: shape (P,) => indices of pairs
    
    Returns scalar BCE loss.
    """
    s_i = pred_scores[i_idx]  # (P,)
    s_j = pred_scores[j_idx]  # (P,)
    
    # label y=1 if true_scores[i]>true_scores[j], else 0
    label = (true_scores[i_idx] > true_scores[j_idx]).float()  # shape (P,)
    
    delta_logit = s_i - s_j  # shape (P,)
    # BCEWithLogitsLoss expects shape (P,) for both input and target
    loss_val = bce_loss_fn(delta_logit, label)  # single scalar
    return loss_val


##########################################################################
# 1) Pairwise BCE Loss
##########################################################################

# Assume we already defined:
#   - bce_loss_fn = nn.BCEWithLogitsLoss()
#   - pairwise_bce_loss(pred_scores, true_scores, i_idx, j_idx)

##########################################################################
# 2) Pairwise Model Training
##########################################################################

def train_pairwise_epoch(model,
                               optimizer,
                               train_dataset,
                               test_dataset,
                               epoch,
                               num_pairs=64,  # how many pairs to sample per iteration
                               lr=1e-4):
    """
    Trains a CrossAttnPairwiseModel (or similar) using pairwise BCE loss
    to encourage correct ranking of contours for ONE epoch.
    
    Every 10 epochs, we also evaluate on 'test_dataset' using the same pairwise loss
    and print the average test loss.
    
    Args:
        model (nn.Module):
            The cross-attention or pairwise model that takes (img_emb, all_g_embs)
            and outputs a score for each contour.
        optimizer (torch.optim.Optimizer):
            The optimizer for training.
        train_dataset (Iterable):
            Each item is a dict for one image with 'img_embedding' and 'cts'.
        test_dataset (Iterable):
            Each item is a dict for one image (for evaluation).
        epoch (int):
            Current epoch index (0-based or 1-based).
        num_pairs (int):
            How many (i, j) pairs to sample from the full set of contours (N ~ 800).
        lr (float):
            Learning rate if needed (unused here but can be used for scheduling).

    Returns:
        (float, float):
            (avg_train_loss, avg_test_loss).
            If we do not evaluate on test this epoch, avg_test_loss = None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    

    epoch_loss = 0.0
    # --------------------
    # Train loop (ONE epoch)
    # --------------------
    for data in train_dataset:
        # 1) Get image + up to 800 contours
 
        img_emb, all_g_embs, all_scores = get_image_and_contours(data)

        # Move to GPU if available
        # shape = (1, 1024, 4, 4, 4)
        img_emb = img_emb.unsqueeze(0).to(device)
        # shape = (N, 1152) for contours, (N,) for scores
        all_g_embs = all_g_embs.to(device)
        all_scores = all_scores.to(device)

        N = all_scores.shape[0]
        
        # 2) We'll sample 'num_pairs' from possible N*N combos
        #    i_idx, j_idx each shape (num_pairs,).
        pair_indices = torch.randint(low=0, high=N*N, size=(num_pairs,), device=device)
        i_idx = pair_indices // N
        j_idx = pair_indices % N
        
        # 3) Predict scores for all N contours => shape (1, N) => squeeze => (N,)

        pred_scores = model(img_emb, all_g_embs.unsqueeze(0)).squeeze(0)  # (N,)

        # 4) Compute pairwise BCE loss
        loss_val = pairwise_bce_loss(pred_scores, all_scores, i_idx, j_idx)

        # 5) Backprop
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        epoch_loss += loss_val.item()

    avg_train_loss = epoch_loss / max(len(train_dataset), 1)
    print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.6f}")

    # -----------------------------------
    # Evaluate on test_dataset every 10 epochs
    # -----------------------------------
    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for test_data in test_dataset:
                # get test image & contours
                img_emb, all_g_embs, all_scores = get_image_and_contours(test_data)
                
                img_emb = img_emb.unsqueeze(0).to(device)
                all_g_embs = all_g_embs.to(device)
                all_scores = all_scores.to(device)
                
                N_test = all_scores.shape[0]
                # sample pairs
                pair_indices_test = torch.randint(low=0, high=N_test*N_test,
                                                  size=(num_pairs,),
                                                  device=device)
                i_idx_test = pair_indices_test // N_test
                j_idx_test = pair_indices_test % N_test
                
                # predict
                pred_test_scores = model(img_emb, all_g_embs.unsqueeze(0)).squeeze(0)
                
                # pairwise BCE
                loss_test_val = pairwise_bce_loss(pred_test_scores,
                                                  all_scores,
                                                  i_idx_test,
                                                  j_idx_test)
                total_test_loss += loss_test_val.item()
        
        test_loss = total_test_loss / max(len(test_dataset), 1)
        print(f" # # # # # # Test Loss | Epoch {epoch+1} # # # # # #  :  {test_loss:.6f}")

    return avg_train_loss, test_loss





##############################################################################
# 1) Cross-Attention Model
##############################################################################

class CrossAttnListwiseModel(nn.Module):
    """
    A simple cross-attention model that:
      - Flattens the image embedding (1024,4,4,4) -> (64,1024) tokens.
      - Projects them to 'hidden_dim'.
      - Projects each contour embedding to 'hidden_dim'.
      - Applies multi-head attention with the K contours as queries,
        the 64 image tokens as key/value.
      - Outputs a score for each of the K contours.
    """
    def __init__(self, 
                 img_channels=1024,  # 1024 channels
                 contour_dim=1152,
                 hidden_dim=256,
                 n_heads=4):
        super().__init__()
        
        self.img_channels = img_channels
        self.hidden_dim = hidden_dim
        
        # Linear projection for image tokens (1024 -> hidden_dim)
        self.img_token_proj = nn.Linear(img_channels, hidden_dim)
        
        # Linear projection for contour embeddings (1152 -> hidden_dim)
        self.contour_proj = nn.Linear(contour_dim, hidden_dim)
        
        # Multi-head attention:
        # We'll treat the K contours as a "sequence" of queries (batch_first=True).
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                num_heads=n_heads,
                                                batch_first=True)
        
        # Small feed-forward after attention
        self.post_attn_ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Final scoring head => scalar
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, img_emb, contour_embs):
        """
        img_emb:      shape (B, 1024, 4, 4, 4) 
                      We'll flatten to (B,64,1024)
        contour_embs: shape (B, K, 1152) 
                      We'll produce a score for each contour => shape (B,K)
        """
        B, C, D, H, W = img_emb.shape
        # Flatten image => (B, 64, 1024)
        img_tokens = img_emb.view(B, C, -1).transpose(1,2)  # (B,64,1024)
        
        # Project image tokens => (B,64,hidden_dim)
        img_tokens_proj = self.img_token_proj(img_tokens)
        
        # Project contour embeddings => (B,K,hidden_dim)
        B2, K, _ = contour_embs.shape
        assert B2 == B, "Image batch size must match contour batch size"
        contour_q = self.contour_proj(contour_embs)  # (B,K,hidden_dim)
        
        # Cross-attention:
        # Query = (B,K,hidden_dim), Key=Value=(B,64,hidden_dim)
        # Output => (B,K,hidden_dim)
        attn_output, _ = self.cross_attn(query=contour_q, 
                                         key=img_tokens_proj, 
                                         value=img_tokens_proj)
        
        # Optional feed-forward
        fused = self.post_attn_ff(attn_output)  # (B,K,hidden_dim)
        
        # Final scoring => (B,K,1)
        scores = self.score_head(fused).squeeze(-1)  # (B,K)
        return scores

##############################################################################
# 2) LambdaRank-Like Loss (Simplified)
##############################################################################

def lambda_rank_loss(pred_scores, true_scores, alpha=1.0):
    """
    A simplified LambdaRank-style loss that:
      1) Sorts items by predicted score to find predicted ranks.
      2) Approximates ΔNDCG for each pair (i,j).
      3) Accumulates "lambda" gradient signals.
      4) Defines a loss whose gradient w.r.t. pred_scores approximates that.

    pred_scores: (K,) predicted scores
    true_scores: (K,) ground-truth relevance in [0,1]
    alpha: scale factor for logistic function

    Returns: A scalar loss that we backprop.
    """
    K = pred_scores.shape[0]
    device = pred_scores.device
    
    # 1) Sort items by predicted score, descending => rank them
    sorted_idx = torch.argsort(pred_scores, descending=True)
    # ranks[i] = predicted rank (0-based) of item i
    ranks = torch.empty_like(sorted_idx, dtype=torch.float)
    ranks[sorted_idx] = torch.arange(K, device=device, dtype=torch.float)
    
    # We'll accumulate a "gradient" for each item i in 'lambda_i'
    lambda_i = torch.zeros(K, device=device)
    
    # 2) Double loop is O(K^2). For large K, you should sample pairs or optimize.
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            
            # ground-truth difference
            dy = true_scores[i] - true_scores[j]
            if dy == 0:
                continue  # no preference between i and j

            # predicted difference
            diff_ij = pred_scores[i] - pred_scores[j]
            
            # logistic factor
            # if y_i > y_j but pred_scores[i] < pred_scores[j], we want to push i up, j down
            rho = 1.0 / (1.0 + torch.exp(alpha * diff_ij))  # sigma(-alpha * diff_ij)
            
            # approximate ΔNDCG from swapping i,j => depends on predicted ranks
            rank_i = ranks[i].item() + 1.0  # 1-based rank
            rank_j = ranks[j].item() + 1.0
            # simplistic approach to magnitude
            delta_ndcg = torch.abs((1.0 / torch.log2(torch.tensor(rank_i+1.0, device=device)))
                                 - (1.0 / torch.log2(torch.tensor(rank_j+1.0, device=device))))
            
            # sign => + if y_i>y_j, - if y_i<y_j
            sign = 1.0 if dy > 0 else -1.0
            
            # accumulate
            lambda_i[i] += sign * delta_ndcg * rho
    
    # 3) We want the gradient dL/d pred_scores[i] = - lambda_i[i].
    #    So define L = -sum_i( lambda_i[i] * pred_scores[i] ).
    #    Then dL/d pred_scores[i] = - lambda_i[i].
    #    Minimizing L => moves pred_scores[i] in direction of +lambda_i[i].
    
    loss_val = - (lambda_i * pred_scores).sum()
    return loss_val

##############################################################################
# 3) Example Training Loop
##############################################################################

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

# Let's define an example training loop:

def train_lambdarank_epoch(model, optimizer,
                           train_dataset, test_dataset,
                           epoch,
                           K=64,   # subset of contours per iteration
                           lr=1e-4):
    """
    Trains the given 'model' for ONE epoch using a LambdaRank-style loss.
    Then, if (epoch+1) % 10 == 0, does a test evaluation pass.

    Args:
        model (nn.Module): The cross-attention ranking model.
        optimizer (torch.optim.Optimizer): The optimizer.
        train_dataset (iterable/list): Contains training data items (dict format).
        test_dataset (iterable/list): Contains test data items (dict format).
        epoch (int): Current epoch index (0-based or 1-based).
        K (int): How many contours we sample from each image in training.
        lr (float): Learning rate (if needed to adjust dynamically, but unused here).
    
    Returns:
        (float, float): A tuple of (train_loss, test_loss).
                        test_loss is None if not doing evaluation this epoch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    
    epoch_loss = 0.0
    
    # --------------------
    # Training (one pass over train_dataset)
    # --------------------
    for data in train_dataset:
        # 1) get image & up to 800 contours
        img_emb, all_g_embs, all_scores = get_image_and_contours(data)
        
        # move to GPU if available
        img_emb = img_emb.unsqueeze(0).to(device)  # shape (1,1024,4,4,4)
        all_g_embs = all_g_embs.to(device)         # shape (N,1152), typically N=800
        all_scores = all_scores.to(device)         # shape (N,) in [0,1]
        
        # 2) sample K from N
        N = all_scores.shape[0]
        if N <= K:
            chosen_idxs = torch.arange(N, device=device)
        else:
            chosen_idxs = torch.randperm(N, device=device)[:K]
        
        c_emb_subset = all_g_embs[chosen_idxs]   # (K,1152)
        score_subset = all_scores[chosen_idxs]   # (K,)
        
        # reshape for model => (B=1, K, 1152)
        c_emb_subset = c_emb_subset.unsqueeze(0)
        
        # 3) forward => predicted scores => (1,K)
        pred_scores = model(img_emb, c_emb_subset).squeeze(0)  # => (K,)
        
        # 4) compute lambdaRank-like loss
        loss_val = lambda_rank_loss(pred_scores, score_subset)
        
        # 5) backprop
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        epoch_loss += loss_val.item()
    
    avg_train_loss = epoch_loss / max(len(train_dataset), 1)
    print(f"[Epoch {epoch+1}] Train Loss = {avg_train_loss:.6f}")
    
    # --------------------
    # Evaluation every 10 epochs
    # --------------------
    test_loss = None
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            eval_loss = 0.0
            for test_data in test_dataset:
                # get image & contours
                img_emb, all_g_embs, all_scores = get_image_and_contours(test_data)
                
                img_emb = img_emb.unsqueeze(0).to(device)
                all_g_embs = all_g_embs.to(device)
                all_scores = all_scores.to(device)
                
                # sample K for test (or use all if memory allows)
                N_test = all_scores.shape[0]
                if N_test <= K:
                    chosen_test = torch.arange(N_test, device=device)
                else:
                    chosen_test = torch.randperm(N_test, device=device)[:K]
                
                c_emb_subset_test = all_g_embs[chosen_test].unsqueeze(0)  # (1,K,1152)
                score_subset_test = all_scores[chosen_test]               # (K,)
                
                pred_scores_test = model(img_emb, c_emb_subset_test).squeeze(0)  # (K,)
                loss_val_test = lambda_rank_loss(pred_scores_test, score_subset_test)
                eval_loss += loss_val_test.item()
            
            test_loss = eval_loss / max(len(test_dataset), 1)
            print(f" # # # # # # Test Loss | Epoch {epoch+1} # # # # # #  :  {test_loss:.6f}")
    
    return avg_train_loss, test_loss
