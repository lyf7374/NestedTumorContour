import torch

def furthest_point_sample_cpu(xyz, npoint):
    """
    Pure Python (CPU) fallback for furthest point sampling.
    This is O(B * N * npoint) and can be slow for large N.
    
    Input:
        xyz: (B, N, 3) batch of point clouds
        npoint: int, number of samples
        
    Return:
        idx: (B, npoint) indices of sampled points
    """
    B, N, _ = xyz.shape
    device = xyz.device

    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distances = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    for i in range(npoint):
        idx[:, i] = farthest  # record the chosen point
        centroids = xyz[torch.arange(B, device=device), farthest, :].unsqueeze(1)  # (B,1,3)
        dist = torch.sum((xyz - centroids) ** 2, dim=-1)                           # (B,N)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = torch.max(distances, dim=1)[1]

    return idx
