import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from models.DCMModel import all_pair_indices, infiltration_loss_multi, get_image_and_contours_rank
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_one_sample(emb_g, d_scores, n_tc=100, n_inf=300, tau=5.0,w_func=None, margin=0.0,detail=False):
    """
    Compute losses for one sample given the global embeddings and d_scores.
    
    Parameters:
      emb_g: Tensor of shape (P_test, hidden_dim) - the computed embeddings.
      d_scores: Tensor containing the d_scores.
      n_tc: Number of tumor core embeddings.
      n_inf: Number of infiltration embeddings.
      tau: Temperature parameter for infiltration loss.
      w_func: Weighting function used by infiltration_loss_multi.
      
    Returns:
      pos_loss_tc: Positive loss for tumor core.
      pos_loss_ht: Positive loss for healthy tissue.
      loss_inf: Infiltration loss.
      smooth_loss: Smoothness loss over the infiltration zone.
    """
    # Partition the embeddings
    e_tc = emb_g[:n_tc]              # (n_tc, hidden_dim)
    e_inf = emb_g[n_tc : n_tc+n_inf]   # (n_inf, hidden_dim)
    e_ht = emb_g[n_tc+n_inf :]         # (remaining, hidden_dim)
    
    # Compute pairwise loss for tumor core embeddings (TC)
    i_idx, j_idx = all_pair_indices(e_tc.size(0))
    emb_i = e_tc[i_idx]
    emb_j = e_tc[j_idx]
    cos_ij = F.cosine_similarity(emb_i, emb_j, dim=-1)  # shape: (num_pairs,)
    pos_loss_tc = ((1.0 - cos_ij) ** 2).mean()

    # Compute pairwise loss for healthy tissue embeddings (HT)
    i_idx, j_idx = all_pair_indices(e_ht.size(0))
    emb_i = e_ht[i_idx]
    emb_j = e_ht[j_idx]
    cos_ij = F.cosine_similarity(emb_i, emb_j, dim=-1)
    pos_loss_ht = ((1.0 - cos_ij) ** 2).mean()

    pos_loss = (pos_loss_ht + pos_loss_tc)/2


    i_idx = torch.arange(e_tc.size(0), device=device)
    j_idx = torch.arange(e_ht.size(0), device=device)
    # Create cross-product indices
    i_idx = i_idx.unsqueeze(1).expand(-1, e_ht.size(0)).reshape(-1)
    j_idx = j_idx.unsqueeze(0).expand(e_tc.size(0), -1).reshape(-1)
    emb_i = e_tc[i_idx]  # (n_tc*n_ht, hidden_dim)
    emb_j = e_ht[j_idx]  # (n_tc*n_ht, hidden_dim)
    cos_ij = F.cosine_similarity(emb_i, emb_j, dim=-1)

    margin_diff = F.relu(cos_ij - margin)
    neg_loss = (margin_diff ** 2).mean()



    # Compute smoothness loss for the infiltration zone embeddings (INF)
    sum_val = torch.tensor(0.0, device=device, dtype=e_inf.dtype)
    for i in range(e_inf.size(0) - 1):
        # Unsqueeze to ensure cosine_similarity sees 2D tensors
        cos_ii1 = F.cosine_similarity(e_inf[i].unsqueeze(0), e_inf[i+1].unsqueeze(0), dim=-1)
        sum_val = sum_val + (1.0 - cos_ii1) ** 2
    smooth_loss = sum_val / (e_inf.size(0) - 1)

    # Ensure d_scores is valid and extract infiltration zone scores
    if d_scores is not None:
        d_inf = d_scores[n_tc:n_tc+n_inf]
    else:
        raise ValueError("d_scores must be provided if lambda_inf > 0.")
    
    # Compute infiltration loss using your predefined function
    loss_inf = infiltration_loss_multi(e_inf, d_inf, e_tc, e_ht, tau, w_func,detail=detail)
    
    return pos_loss, neg_loss, loss_inf, smooth_loss

def evaluate_dataset(test_dataset, model, n_tc=100, n_inf=300, tau=5.0, w_func=None,detail=False):
    """
    Loop over the test_dataset, compute losses for each valid sample, and return average losses.
    
    Parameters:
      test_dataset: Iterable dataset (e.g., val_loader)
      model: The model used to compute embeddings.
      n_tc: Number of tumor core embeddings.
      n_inf: Number of infiltration embeddings.
      tau: Temperature parameter for infiltration loss.
      w_func: Weighting function used by infiltration_loss_multi.
      
    Returns:
      A dictionary with average losses.
    """
    total_pos_loss = 0.0
    total_neg_loss = 0.0
    total_loss_inf   = 0.0
    total_smooth_loss = 0.0
    valid_sample_count = 0

    for val_item in tqdm.tqdm(test_dataset):
        # Compute the image embeddings, contour embeddings, and d_scores for the sample.
        image_emb, all_g_embs, d_scores = get_image_and_contours_rank(val_item)
        if len(image_emb.shape) == 4:
            image_emb = image_emb.unsqueeze(0)
        image_emb = image_emb.to(device)
        all_g_embs = all_g_embs.to(device)
        d_scores   = d_scores.to(device)

        # Ensure there are enough embeddings for the given thresholds.
        P_test = all_g_embs.size(0)
        if P_test < (n_tc + n_inf + 1):
            continue

        # Create an image batch matching the number of embeddings.
        image_batch = image_emb.repeat(P_test, 1, 1, 1, 1)
        with torch.no_grad():
            emb_g = model(image_batch, all_g_embs)

        # Compute losses for this sample.
        pos_loss, neg_loss, loss_inf, smooth_loss = test_one_sample(emb_g, d_scores, n_tc, n_inf, tau, w_func,detail=detail)
        
        total_pos_loss += pos_loss.item()
        total_neg_loss += neg_loss.item()
        total_loss_inf   += loss_inf.item()
        total_smooth_loss += smooth_loss.item()
        valid_sample_count += 1

    if valid_sample_count == 0:
        raise ValueError("No valid samples found in the dataset!")

    avg_losses = {
        'avg_pos_loss': total_pos_loss / valid_sample_count,
        'avg_neg_loss': total_neg_loss / valid_sample_count,
        'avg_loss_inf':    total_loss_inf   / valid_sample_count,
        'avg_smooth_loss': total_smooth_loss / valid_sample_count
    }

    print(f"Average positive loss: {avg_losses['avg_pos_loss']:.5f}")
    print(f"Average negative loss: {avg_losses['avg_neg_loss']:.5f}")
    print(f"Average infiltration loss:  {avg_losses['avg_loss_inf']:.5f}")
    print(f"Average smooth loss:        {avg_losses['avg_smooth_loss']:.5f}")


    return avg_losses