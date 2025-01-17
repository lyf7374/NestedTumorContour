import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.PointSupport import PointNetSetAbstraction
from utils.Gsupport import ConvD,ConvU,normalization


def process_labels_and_distances(labels):
#     labels = labels.view(-1)
    processed_labels = torch.zeros_like(labels, dtype=torch.long)
    distances = labels.clone()
    processed_labels[labels == 0.0] = 0  # HT
    processed_labels[labels == 1.0] = 1  # TC
    mask_uncertain = (labels > 0.0) & (labels < 1.0)
    processed_labels[mask_uncertain] = 2  # Uncertain
    # For HT and TC, distances are set to 0.0 and 1.0 respectively
    distances[labels == 0.0] = 0.0
    distances[labels == 1.0] = 1.0
    # Distances for uncertain samples remain as is
    return processed_labels, distances
def generate_indices(samples_per_group=3, num_groups=4,points_per_group=50):
    # Base indices for one group
    base_indices = torch.randperm(points_per_group)[:samples_per_group]
    # Generate indices for all groups and concatenate them
    all_indices = torch.cat([base_indices + i * points_per_group for i in range(num_groups)])
    
    return all_indices

# def contrastive_loss_vectorized(embeddings_i, embeddings_j, labels_i, labels_j, dists_i, dists_j, pair_types):
#     # Compute pairwise distances between embeddings
#     D_ij = torch.norm(embeddings_i - embeddings_j, p=2, dim=2)  # Shape: (num_pairs,)

#     # Initialize y_ij
#     y_ij = torch.zeros_like(D_ij)

#     # Process each pair type
#     mask_positive = pair_types == 0
#     mask_negative = pair_types == 1
#     mask_uncertain_positive = pair_types == 2
#     mask_uncertain_uncertain = pair_types == 3

#     # Certain positive pairs
#     y_ij[mask_positive] = 1.0

#     # Certain negative pairs
#     y_ij[mask_negative] = 0.0

#     # Uncertain-Uncertain pairs
#     y_ij[mask_uncertain_uncertain] = 1 - torch.abs(dists_i[mask_uncertain_uncertain] - dists_j[mask_uncertain_uncertain])

#     # Uncertain and certain pairs
#     y_ij[mask_uncertain_positive] = torch.where(
#         labels_i[mask_uncertain_positive] == 2,
#         torch.where(labels_j[mask_uncertain_positive] == 1, dists_i[mask_uncertain_positive], 1 - dists_i[mask_uncertain_positive]),
#         torch.where(labels_i[mask_uncertain_positive] == 1, dists_j[mask_uncertain_positive], 1 - dists_j[mask_uncertain_positive])
#     )

#     # Compute loss
#     loss = torch.mean((D_ij - (1 - y_ij)) ** 2)

#     return loss
def contrastive_loss_vectorized(embeddings_i, embeddings_j, labels_i, labels_j, dists_i, dists_j, pair_types):
    # Compute pairwise distances between embeddings
    D_ij = torch.norm(embeddings_i - embeddings_j, p=2, dim=1)  # Shape: (num_pairs,)

    # Initialize y_ij
    y_ij = torch.zeros_like(D_ij)

    # Process each pair type
    mask_positive = pair_types == 0
    mask_negative = pair_types == 1
    mask_uncertain_positive = pair_types == 2
    mask_uncertain_uncertain = pair_types == 3

    # Certain positive pairs
    y_ij[mask_positive] = 1.0

    # Certain negative pairs
    y_ij[mask_negative] = 0.0

    # Uncertain-Uncertain pairs
    y_ij[mask_uncertain_uncertain] = 1 - torch.abs(dists_i[mask_uncertain_uncertain] - dists_j[mask_uncertain_uncertain])

    # Uncertain and certain pairs
    y_ij[mask_uncertain_positive] = torch.where(
        labels_i[mask_uncertain_positive] == 2,
        torch.where(
            labels_j[mask_uncertain_positive] == 1,
            dists_i[mask_uncertain_positive],
            1 - dists_i[mask_uncertain_positive]
        ),
        torch.where(
            labels_i[mask_uncertain_positive] == 1,
            dists_j[mask_uncertain_positive],
            1 - dists_j[mask_uncertain_positive]
        )
    )

    # Compute loss
    loss = torch.mean((D_ij - (1 - y_ij)) ** 2)

    return loss

# def extract_pairs(cts_select, labels, dists, mask, i_indices, j_indices, embeddings):
#     batch_size = cts_select.shape[0]

#     # Get the indices where the mask is True
#     valid_pairs = mask.nonzero(as_tuple=False)  # shape [num_valid_pairs_total, 2]
#     batch_idx = valid_pairs[:, 0]
#     pair_idx = valid_pairs[:, 1]

#     # Extract the indices for i and j
#     i_idx = i_indices[batch_idx, pair_idx]  # shape [num_valid_pairs_total]
#     j_idx = j_indices[batch_idx, pair_idx]

#     # Extract cts_select pairs
#     cts_i = cts_select[batch_idx, i_idx, :, :]  # shape [num_valid_pairs_total, 4096, 3]
#     cts_j = cts_select[batch_idx, j_idx, :, :]

#     # Extract labels and distances
#     labels_i = labels[batch_idx, i_idx]
#     labels_j = labels[batch_idx, j_idx]

#     dists_i = dists[batch_idx, i_idx]
#     dists_j = dists[batch_idx, j_idx]

#     # Extract vector_select pairs
#     embed_i =  embeddings[batch_idx, i_idx, :]  # shape [num_valid_pairs_total, 128]
#     embed_j =  embeddings[batch_idx, j_idx, :]


#     # Compute the number of valid pairs per batch
#     num_valid_pairs_per_batch = mask.sum(dim=1).cpu().numpy()  # Shape: [batch_size]
#     max_num_pairs = num_valid_pairs_per_batch[0]  # Since it's the same for all batches

#     print('batch_size, max_num_pairs, embeddings.shape[2] cts_select.shape[2], cts_select.shape[3]',batch_size, max_num_pairs, embeddings.shape[2],cts_select.shape[2], cts_select.shape[3])
#     print('cts_i ,cts_J',cts_i.shape, cts_j.shape)
#     # Reshape outputs to include the batch dimension
#     cts_i = cts_i.view(batch_size, max_num_pairs, cts_select.shape[2], cts_select.shape[3])
#     cts_j = cts_j.view(batch_size, max_num_pairs, cts_select.shape[2], cts_select.shape[3])

#     labels_i = labels_i.view(batch_size, max_num_pairs)
#     labels_j = labels_j.view(batch_size, max_num_pairs)
#     dists_i = dists_i.view(batch_size, max_num_pairs)
#     dists_j = dists_j.view(batch_size, max_num_pairs)
   
#     embed_i = embed_i.view(batch_size, max_num_pairs, embeddings.shape[2])
#     embed_j = embed_j.view(batch_size, max_num_pairs, embeddings.shape[2])

#     return cts_i, cts_j, labels_i, labels_j, dists_i, dists_j,embed_i,embed_j

def extract_pairs_noembed(cts_select, labels, dists, mask, i_indices, j_indices):
    batch_size = cts_select.shape[0]
    cts_i_list = []
    cts_j_list = []
    labels_i_list = []
    labels_j_list = []
    dists_i_list = []
    dists_j_list = []
    batch_indices_list = []

    for b in range(batch_size):
        # Get the indices where the mask is True for this batch
        valid_pairs = mask[b].nonzero(as_tuple=False).squeeze(1)
        num_valid_pairs = valid_pairs.shape[0]
        if num_valid_pairs == 0:
            continue

        # Extract indices for i and j
        i_idx = i_indices[b, valid_pairs]
        j_idx = j_indices[b, valid_pairs]

        # Extract cts_select pairs
        cts_i = cts_select[b, i_idx, :, :]  # shape [num_valid_pairs, 4096, 3]
        cts_j = cts_select[b, j_idx, :, :]

        # Extract labels and distances
        labels_i = labels[b, i_idx]
        labels_j = labels[b, j_idx]
        dists_i = dists[b, i_idx]
        dists_j = dists[b, j_idx]

        # Record batch indices for each pair
        batch_indices = torch.full((num_valid_pairs,), b, dtype=torch.long, device=cts_select.device)

        # Append to lists
        cts_i_list.append(cts_i)
        cts_j_list.append(cts_j)
        labels_i_list.append(labels_i)
        labels_j_list.append(labels_j)
        dists_i_list.append(dists_i)
        dists_j_list.append(dists_j)
        batch_indices_list.append(batch_indices)

    # Concatenate lists
    cts_i = torch.cat(cts_i_list, dim=0)
    cts_j = torch.cat(cts_j_list, dim=0)
    labels_i = torch.cat(labels_i_list, dim=0)
    labels_j = torch.cat(labels_j_list, dim=0)
    dists_i = torch.cat(dists_i_list, dim=0)
    dists_j = torch.cat(dists_j_list, dim=0)
    batch_indices = torch.cat(batch_indices_list, dim=0)

    return cts_i, cts_j, labels_i, labels_j, dists_i, dists_j, batch_indices

def extract_pairs(cts_select, labels, dists, mask, i_indices, j_indices, embeddings):
    batch_size = cts_select.shape[0]
    cts_i_list = []
    cts_j_list = []
    labels_i_list = []
    labels_j_list = []
    dists_i_list = []
    dists_j_list = []
    embed_i_list = []
    embed_j_list = []

    for b in range(batch_size):
        # Get the indices where the mask is True for this batch
        valid_pairs = mask[b].nonzero(as_tuple=False).squeeze(1)
        num_valid_pairs = valid_pairs.shape[0]
        if num_valid_pairs == 0:
            continue

        # Extract indices for i and j
        i_idx = i_indices[b, valid_pairs]
        j_idx = j_indices[b, valid_pairs]

        # Extract cts_select pairs
        cts_i = cts_select[b, i_idx, :, :]  # shape [num_valid_pairs, 4096, 3]
        cts_j = cts_select[b, j_idx, :, :]

        # Extract labels and distances
        labels_i = labels[b, i_idx]
        labels_j = labels[b, j_idx]
        dists_i = dists[b, i_idx]
        dists_j = dists[b, j_idx]

        # Extract embeddings
        embed_i = embeddings[b, i_idx, :]
        embed_j = embeddings[b, j_idx, :]

        # Append to lists
        cts_i_list.append(cts_i)
        cts_j_list.append(cts_j)
        labels_i_list.append(labels_i)
        labels_j_list.append(labels_j)
        dists_i_list.append(dists_i)
        dists_j_list.append(dists_j)
        embed_i_list.append(embed_i)
        embed_j_list.append(embed_j)

    # Concatenate lists
    cts_i = torch.cat(cts_i_list, dim=0)
    cts_j = torch.cat(cts_j_list, dim=0)
    labels_i = torch.cat(labels_i_list, dim=0)
    labels_j = torch.cat(labels_j_list, dim=0)
    dists_i = torch.cat(dists_i_list, dim=0)
    dists_j = torch.cat(dists_j_list, dim=0)
    embed_i = torch.cat(embed_i_list, dim=0)
    embed_j = torch.cat(embed_j_list, dim=0)

    return cts_i, cts_j, labels_i, labels_j, dists_i, dists_j, embed_i, embed_j

class IG_fusion(nn.Module):
    def __init__(self, inC,outC,outS, expS):
        #  eg:（114,114,114,16）-> (1024,64),  inS = 114, inC=16, outS=16, outC=64, expS = 1024
        super(IG_fusion, self).__init__()
               
        self.conv = nn.Conv3d(inC, outC, 7, 7, 1, groups=inC, bias=True)
        self.ln = nn.LayerNorm((outC,outS, outS, outS))
        self.mlp_1 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        self.norm = nn.GELU()
        self.mlp_2 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        
        self.outC= outC
        self.flat1 = nn.Conv1d(outS*outS*outS,expS,1,1,0)
        self.flat2 = nn.Conv1d(expS,expS,1,1,0)
        
        self.map1 = nn.Conv1d(outC*2,outC,1,1,0)
        self.map2 = nn.Conv1d(outC,outC,1,1,0)    
        
    def forward(self, x, y):
        batch_size = x.shape[0]
        #   process pc 
        x = self.conv(x) 
        x = self.ln(x) 
        x = self.mlp_1(x) 
        x = self.norm(x) 
        x = self.mlp_2(x) 

        #   flatten
        # x = x.view(1,-1,self.outC)
        x = x.view(batch_size, -1, self.outC)
        x = self.flat1(x)
        x = self.norm(self.flat2(x))
        x = x.permute(0,2,1)

        #   Concat and fusion

        z = torch.cat([y,x],1)  #   (expS,outC*2)      

        z = self.map1(z)
        z = self.norm(self.map2(z))      
        
        return z
    
class PCModel_wr_v3(nn.Module):
    def __init__(self,n_layers=5,c=1, n=8, dropout=0.1, norm='bn',n_pc=4096):
        super(PCModel_wr_v3, self).__init__()
        self.sa1 = PointNetSetAbstraction(n_pc, 0.05, 32, 3 + 3, [4, 4, 8], False)
        self.sa2 = PointNetSetAbstraction(1024, 0.1, 16, 8 + 3, [8, 8, 16], False)
        self.sa3 = PointNetSetAbstraction(256, 0.2, 8,   16 + 3, [16, 16, 32], False)
        self.sa4 = PointNetSetAbstraction(64, 0.4, 4,    32 + 3, [32, 32, 64], False)
        self.sa5 = PointNetSetAbstraction(16, 0.8, 2,    64 + 3, [64, 64, 128], False)
        
        self.IG1 = IG_fusion(8,8,27,n_pc)
        self.IG2 = IG_fusion(16,16,14,1024)
        self.IG3 = IG_fusion(32,32,7,256)
        self.IG4 = IG_fusion(64,64,3,64)
        self.IG5 = IG_fusion(128,128,2,16)

    def forward(self, xyz, ys):

        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        # l0_r = xyz[:, 3:4, :]  # Radius
        
        fused_features = []
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l1_points = self.IG1(ys[0] ,l1_points)
            
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2(ys[1] ,l2_points)
            
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3(ys[2] ,l3_points)
            
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4(ys[3] ,l4_points)
        
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l5_points = self.IG5(ys[4] ,l5_points)
    
        for ele in [l1_points ,l2_points ,l3_points ,l4_points ,l5_points]:
            fused_features.append(ele)
    
        return fused_features
    

class FeatureAggregator(nn.Module):
    def __init__(self, channel_dims, embedding_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(len(channel_dims)))
        self.projection_heads = nn.ModuleList([
            nn.Linear(ch_dim, embedding_dim) for ch_dim in channel_dims
        ])

    def forward(self, pooled_features_list):
        projected_features = []
        for i, features in enumerate(pooled_features_list):
            projected = self.projection_heads[i](features)  # Shape: (batch_size, embedding_dim)
            projected_features.append(projected)
        # Stack along a new dimension (num_levels)
        stacked_features = torch.stack(projected_features, dim=2)  # Shape: (batch_size, embedding_dim, num_levels)
        # Apply softmax to the weights and expand dimensions to match
        weights = F.softmax(self.weights, dim=0)  # Shape: (num_levels,)
        weights = weights.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, num_levels)
        # Apply weighted sum
        weighted_features = (stacked_features * weights).sum(dim=2)  # Shape: (batch_size, embedding_dim)
        return weighted_features
    
class DCM_model(nn.Module):
    def __init__(self, n_layers=5, c=1, n=8, padding_list=[0, 0, 0, 0, 0, 0, 0],
                 norm='bn', dropout=0.1):
        super( DCM_model, self).__init__()

        self.middle_channel = 2 ** (n_layers) * n
        self.dropout = dropout
        self.padding_list = padding_list
        self.n_layers = n_layers
        self.pooling = nn.MaxPool3d(2, 2, 0)

        # Down sampling layers
        self.convd_list = nn.ModuleList([
            ConvD(c, n, self.dropout, norm, first=True) if i == 0 else
            ConvD(2 ** (i - 1) * n, 2 ** i * n, self.dropout, norm,
                  padding=self.padding_list[i - 1])
            for i in range(n_layers + 1)
        ])
        # Up sampling layers
        self.convu_list = nn.ModuleList([
            ConvU(2 ** (i + 1) * n, self.dropout, norm,
                  first=True if i == n_layers - 1 else False, padding=self.padding_list[i])
            for i in reversed(range(n_layers))
        ])

        self.pc_evolve = PCModel_wr_v3()

        # Initialize the weights for both sets of layers
        self._initialize_weights()

        # FeatureAggregator and projection heads initialization
        channel_dims = [8, 16, 32, 64, 128]  # Channels at each level
        embedding_dim = 128  # Dimension of the embedding
        self.feature_aggregator = FeatureAggregator(channel_dims, embedding_dim)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, path):
        # Load the model
        saved_state_dict = torch.load(path, map_location='cpu')

        # Prepare the current model's state dictionary
        model_state_dict = self.state_dict()

        # Track which parameters are loaded
        loaded_params = set()

        # Filter out unnecessary keys from the loaded state dictionary
        for name in list(saved_state_dict.keys()):  # Use list to copy keys
            if name in model_state_dict:
                if model_state_dict[name].size() == saved_state_dict[name].size():
                    loaded_params.add(name)
                else:
                    print(f"Skipping {name} due to size mismatch: Model expects {model_state_dict[name].size()}, "
                          f"loaded state provides {saved_state_dict[name].size()}.")
                    saved_state_dict.pop(name)
            else:
                pass  # Key not in current model's state_dict

        # Update current model's state dictionary with the filtered loaded state dictionary
        model_state_dict.update(saved_state_dict)

        # Load the updated state dictionary back to the model
        self.load_state_dict(model_state_dict, strict=False)

        return loaded_params

    def forward(self, x, coors_g1, coors_g2, labels, dists):
        device = x.device

        # Process input through encoder-decoder to get multi-level features
        y_initial = self.process_layers(self.convd_list, self.convu_list, x)

        # Generate point cloud features for both groups
        f_out1 = self.pc_evolve(coors_g1, y_initial)  # List of features at each level
        f_out2 = self.pc_evolve(coors_g2, y_initial)  # List of features at each level

        # Compute the customized contrastive loss
        loss = self.compute_custom_contrastive_loss(f_out1, f_out2, labels, dists)

        return loss

    def process_layers(self, convd_list, convu_list, x):
        xs = []
        ys = []

        for i, conv in enumerate(convd_list):
            x = conv(x)
            if i != len(convd_list) - 1:
                xs.append(x)

        y = x

        for i, convu in enumerate(convu_list):
            y = convu(y, xs[::-1][i])
            ys.append(y)

        return ys[::-1]

    def global_pooling(self, features):
        # Apply global max pooling over the 'num_points' dimension
        pooled_features = torch.max(features, dim=2)[0]  # Shape: (batch_size, channels_n)
        return pooled_features


    def compute_supervised_contrastive_loss(self, f_out1, f_out2, labels, temperature=0.5):
        # Apply global pooling to each level
        pooled_features1 = [self.global_pooling(f) for f in f_out1]
        pooled_features2 = [self.global_pooling(f) for f in f_out2]

        # Aggregate features using FeatureAggregator
        embedding1 = self.feature_aggregator(pooled_features1)  # Shape: (batch_size, embedding_dim)
        embedding2 = self.feature_aggregator(pooled_features2)  # Shape: (batch_size, embedding_dim)

        # Normalize embeddings
        embedding1 = F.normalize(embedding1, p=2, dim=1)
        embedding2 = F.normalize(embedding2, p=2, dim=1)

        # Compute supervised contrastive loss
        loss = self.supervised_contrastive_loss_function(embedding1, embedding2, labels, temperature)
        return loss


    def supervised_contrastive_loss_function(self, embedding1, embedding2, labels, temperature=0.5):
        """
        embedding1: Tensor of shape (batch_size, embedding_dim)
        embedding2: Tensor of shape (batch_size, embedding_dim)
        labels: Tensor of shape (batch_size,) with values 1 for positive pairs, 0 for negative pairs
        """
        batch_size = embedding1.size(0)
        device = embedding1.device

        # Concatenate embeddings
        embeddings = torch.cat([embedding1, embedding2], dim=0)  # Shape: (2*batch_size, embedding_dim)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # Shape: (2*batch_size, 2*batch_size)
        similarity_matrix /= temperature

        # Create labels matrix
        labels = labels.view(-1, 1)
        labels = torch.cat([labels, labels], dim=0)  # Shape: (2*batch_size, 1)
        labels_matrix = torch.eq(labels, labels.T).float().to(device)  # Shape: (2*batch_size, 2*batch_size)

        # Mask to remove self-comparisons
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        labels_matrix = labels_matrix.masked_fill(mask, 0)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))

        # Compute log-softmax over similarity matrix
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # Compute the loss for positive pairs
        mean_log_prob_pos = (labels_matrix * log_prob).sum(1) / labels_matrix.sum(1)

        # Loss is the negative of the mean log-likelihood of positives
        loss = -mean_log_prob_pos.mean()
        return loss
    def pairwise_contrastive_loss(self, embedding1, embedding2, labels, margin=1.0):
        """
        embedding1: Tensor of shape (batch_size, embedding_dim)
        embedding2: Tensor of shape (batch_size, embedding_dim)
        labels: Tensor of shape (batch_size,) with values 1 for positive pairs, 0 for negative pairs
        """
        # Compute pairwise distances
        distances = F.pairwise_distance(embedding1, embedding2)

        # Compute loss
        loss_pos = labels * distances.pow(2)
        loss_neg = (1 - labels) * F.relu(margin - distances).pow(2)
        loss = 0.5 * (loss_pos + loss_neg).mean()
        return loss
    def compute_pairwise_contrastive_loss(self, f_out1, f_out2, labels, margin=1.0):
        # Apply global pooling to each level
        pooled_features1 = [self.global_pooling(f) for f in f_out1]
        pooled_features2 = [self.global_pooling(f) for f in f_out2]

        # Aggregate features using FeatureAggregator
        embedding1 = self.feature_aggregator(pooled_features1)  # Shape: (batch_size, embedding_dim)
        embedding2 = self.feature_aggregator(pooled_features2)  # Shape: (batch_size, embedding_dim)

        # Normalize embeddings (optional)
        # embedding1 = F.normalize(embedding1, p=2, dim=1)
        # embedding2 = F.normalize(embedding2, p=2, dim=1)

        # Compute pairwise contrastive loss
        loss = self.pairwise_contrastive_loss(embedding1, embedding2, labels, margin)
        return loss
    def compute_custom_contrastive_loss(self, f_out1, f_out2, labels, dists, margin=1.0):
        # Apply global pooling to each level
        pooled_features1 = [self.global_pooling(f) for f in f_out1]
        pooled_features2 = [self.global_pooling(f) for f in f_out2]

        # Aggregate features using FeatureAggregator
        embedding1 = self.feature_aggregator(pooled_features1)  # Shape: (batch_size, embedding_dim)
        embedding2 = self.feature_aggregator(pooled_features2)  # Shape: (batch_size, embedding_dim)

        # Normalize embeddings (optional)
        # embedding1 = F.normalize(embedding1, p=2, dim=1)
        # embedding2 = F.normalize(embedding2, p=2, dim=1)

        # Compute the loss
        loss = self.distance_aware_contrastive_loss(embedding1, embedding2, labels, dists, margin)
        return loss
    def distance_aware_contrastive_loss(self, embedding1, embedding2, labels, dists, margin=1.0):
        """
        embedding1: Tensor of shape (batch_size, embedding_dim)
        embedding2: Tensor of shape (batch_size, embedding_dim)
        labels: Tensor of shape (batch_size,) with values 1 for positive pairs, 0 for negative pairs
        dists: Tensor of shape (batch_size,) containing the distance d_i for each pair
        """
        # Compute pairwise distances between embeddings
        distances = F.pairwise_distance(embedding1, embedding2)

        # Compute losses for positive pairs
        loss_pos = labels * (distances - dists).pow(2)

        # Compute losses for negative pairs
        loss_neg = (1 - labels) * F.relu(margin + dists - distances).pow(2)

        # Total loss
        loss = 0.5 * (loss_pos + loss_neg).mean()
        return loss
    def get_embedding(self, x, coors_g):
        # Process input through encoder-decoder to get multi-level features
        y_initial = self.process_layers(self.convd_list, self.convu_list, x)

        # Generate point cloud features
        f_out = self.pc_evolve(coors_g, y_initial)  # List of features at each level

        # Apply global pooling to each level
        pooled_features = [self.global_pooling(f) for f in f_out]

        # Aggregate features using FeatureAggregator
        embedding = self.feature_aggregator(pooled_features)  # Shape: (batch_size, embedding_dim)

        # Normalize embeddings (optional but recommended)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
    def inference(self, x, coors_g1,coors_g2):
        embedding1 = self.get_embedding(x, coors_g1)
        embedding2 = self.get_embedding(x, coors_g2)

        # Compute Euclidean distance
        distance = F.pairwise_distance(embedding1, embedding2)

        # Or compute cosine similarity
        cosine_similarity = F.cosine_similarity(embedding1, embedding2)
        return cosine_similarity
    
    


class DCM_encoder(nn.Module):
    def __init__(self, n_layers=5, c=1, n=8, padding_list=[0, 0, 0, 0, 0, 0, 0],
                 norm='bn', dropout=0.1):
        super( DCM_encoder, self).__init__()

        self.middle_channel = 2 ** (n_layers) * n
        self.dropout = dropout
        self.padding_list = padding_list
        self.n_layers = n_layers
        self.pooling = nn.MaxPool3d(2, 2, 0)

        # Down sampling layers
        self.convd_list = nn.ModuleList([
            ConvD(c, n, self.dropout, norm, first=True) if i == 0 else
            ConvD(2 ** (i - 1) * n, 2 ** i * n, self.dropout, norm,
                  padding=self.padding_list[i - 1])
            for i in range(n_layers + 1)
        ])
        # Up sampling layers
        self.convu_list = nn.ModuleList([
            ConvU(2 ** (i + 1) * n, self.dropout, norm,
                  first=True if i == n_layers - 1 else False, padding=self.padding_list[i])
            for i in reversed(range(n_layers))
        ])

        self.pc_evolve = PCModel_wr_v3()

        # Initialize the weights for both sets of layers
        self._initialize_weights()

        # FeatureAggregator and projection heads initialization
        channel_dims = [8, 16, 32, 64, 128]  # Channels at each level
        embedding_dim = 128  # Dimension of the embedding
        self.feature_aggregator = FeatureAggregator(channel_dims, embedding_dim)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, path):
        # Load the model
        saved_state_dict = torch.load(path, map_location='cpu')

        # Prepare the current model's state dictionary
        model_state_dict = self.state_dict()

        # Track which parameters are loaded
        loaded_params = set()

        # Filter out unnecessary keys from the loaded state dictionary
        for name in list(saved_state_dict.keys()):  # Use list to copy keys
            if name in model_state_dict:
                if model_state_dict[name].size() == saved_state_dict[name].size():
                    loaded_params.add(name)
                else:
                    print(f"Skipping {name} due to size mismatch: Model expects {model_state_dict[name].size()}, "
                          f"loaded state provides {saved_state_dict[name].size()}.")
                    saved_state_dict.pop(name)
            else:
                pass  # Key not in current model's state_dict

        # Update current model's state dictionary with the filtered loaded state dictionary
        model_state_dict.update(saved_state_dict)

        # Load the updated state dictionary back to the model
        self.load_state_dict(model_state_dict, strict=False)

        return loaded_params

    def process_layers(self, convd_list, convu_list, x):
        xs = []
        ys = []

        for i, conv in enumerate(convd_list):
            x = conv(x)
            if i != len(convd_list) - 1:
                xs.append(x)

        y = x

        for i, convu in enumerate(convu_list):
            y = convu(y, xs[::-1][i])
            ys.append(y)

        return ys[::-1]

    def global_pooling(self, features):
        # Apply global max pooling over the 'num_points' dimension
        pooled_features = torch.max(features, dim=2)[0]  # Shape: (batch_size, channels_n)
        return pooled_features



    def forward(self, x, coors_g):
        batch, num_samples, p_dim, num_points = coors_g.shape  # 

        # Process input through encoder-decoder to get multi-level features
        y_initial = self.process_layers(self.convd_list, self.convu_list, x)

        embeddings = []  # Initialize a list to collect embeddings

        for sample in range(num_samples):
            coors_gs = coors_g[:, sample, :, :]  # Extract coordinates for the current sample
          
            # Generate point cloud features
            f_out = self.pc_evolve(coors_gs, y_initial)  # List of features at each level

            # Apply global pooling to each level
            pooled_features = [self.global_pooling(f) for f in f_out]

            # Aggregate features using FeatureAggregator
            embedding = self.feature_aggregator(pooled_features)  # Shape: (batch_size, embedding_dim)

            # Normalize embeddings (optional but recommended)
            embedding = F.normalize(embedding, p=2, dim=1)

            embeddings.append(embedding)  # Append the embedding to the list

        # Stack all embeddings along a new dimension (e.g., dimension 1)
        # Resulting shape: (batch_size, num_samples, embedding_dim)
        stacked_embedding = torch.stack(embeddings, dim=1)

        return stacked_embedding
    

class CustomFusionLayer(nn.Module):
    def __init__(self, y_dim, x_dim, out_dim):
        super(CustomFusionLayer, self).__init__()
        self.linear_y = nn.Linear(y_dim, out_dim)
        self.linear_x = nn.Linear(x_dim, out_dim)
        self.activation = nn.GELU()

    def forward(self, y, x):
        # y: (N, y_dim)
        # x: (1, x_dim)
        y_mapped = self.linear_y(y)     # Shape: (N, out_dim)
        x_mapped = self.linear_x(x)     # Shape: (1, out_dim)
        z = y_mapped + x_mapped         # Broadcasting addition: (N, out_dim) + (1, out_dim)
        z = self.activation(z)          # Shape: (N, out_dim)
        return z

class IG_fusion_v2(nn.Module):
    def __init__(self, inC, outC, outS, expS, point_feature_dim):
        super(IG_fusion_v2, self).__init__()
        self.conv = nn.Conv3d(inC, outC, 7, 7, 1, groups=inC, bias=True)
        self.ln = nn.LayerNorm((outC, outS, outS, outS))
        self.mlp_1 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        self.norm = nn.GELU()
        self.mlp_2 = nn.Conv3d(outC, outC, 1, 1, 0, bias=True)
        
        self.outC = outC
        self.flat1 = nn.Conv1d(outS * outS * outS, expS, 1, 1, 0)
        self.flat2 = nn.Conv1d(expS, expS, 1, 1, 0)
        
        # Replace map1 with a custom fusion layer
        self.fusion = CustomFusionLayer(point_feature_dim, expS, outC)
        
        self.map2 = nn.Conv1d(outC, outC, 1, 1, 0)    

    def process_image(self, x):
        # x: image features of shape (1, inC, H, W, D)
        x = self.conv(x)
        x = self.ln(x)
        x = self.mlp_1(x)
        x = self.norm(x)
        x = self.mlp_2(x)
        x = x.view(1, -1, self.outC)
        x = self.flat1(x)
        x = self.norm(self.flat2(x))
        # Flatten x to shape (1, feature_dim)
        x = x.view(1, -1)
        return x  # Shape: (1, feature_dim)

    def fuse(self, x, y):
        # x: processed image features of shape (1, feature_dim)
        # y: point cloud features of shape (N, point_feature_dim)
        z = self.fusion(y, x)
        z = self.norm(self.map2(z))
        return z  # Shape: (N, outC)

class PCModel_wr_v4(nn.Module):
    def __init__(self, n_pc=4096):
        super(PCModel_wr_v4, self).__init__()
        # Define the PointNet Set Abstraction layers
        self.sa1 = PointNetSetAbstraction(npoint=n_pc, radius=0.05, nsample=32, in_channel=3 + 3, mlp=[4, 4, 8], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=1024, radius=0.1, nsample=16, in_channel=8 + 3, mlp=[8, 8, 16], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=8, in_channel=16 + 3, mlp=[16, 16, 32], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=64, radius=0.4, nsample=4, in_channel=32 + 3, mlp=[32, 32, 64], group_all=False)
        self.sa5 = PointNetSetAbstraction(npoint=16, radius=0.8, nsample=2, in_channel=64 + 3, mlp=[64, 64, 128], group_all=False)
        
        # Initialize IG_fusion_v2 with appropriate dimensions
        # You need to specify the correct dimensions for `point_feature_dim` and `expS`
        # Assuming expS is the feature dimension after `process_image`
        # For simplicity, let's define expS = outC
        point_feature_dims = [8, 16, 32, 64, 128]
        self.IG1 = IG_fusion_v2(inC=8, outC=8, outS=27, expS=8, point_feature_dim=8)
        self.IG2 = IG_fusion_v2(inC=16, outC=16, outS=14, expS=16, point_feature_dim=16)
        self.IG3 = IG_fusion_v2(inC=32, outC=32, outS=7, expS=32, point_feature_dim=32)
        self.IG4 = IG_fusion_v2(inC=64, outC=64, outS=3, expS=64, point_feature_dim=64)
        self.IG5 = IG_fusion_v2(inC=128, outC=128, outS=2, expS=128, point_feature_dim=128)

    def forward(self, xyz, ys):
        # xyz: Shape (1, N, 3, num_points)
        batch_size, N, _, num_points = xyz.shape
        l0_points = xyz.view(-1, 3, num_points)  # Shape: (N, 3, num_points)
        l0_xyz = l0_points

        # Process image features once
        x1 = self.IG1.process_image(ys[0])  # Shape: (1, feature_dim_img1)
        x2 = self.IG2.process_image(ys[1])  # Similarly for x2, x3, x4, x5
        x3 = self.IG3.process_image(ys[2])
        x4 = self.IG4.process_image(ys[3])
        x5 = self.IG5.process_image(ys[4])

        print('lo ',l0_xyz.shape)
        # Level 1
        l1_xyz, l1_points = self.sa1(l0_xyz, None)  # l1_points: (N, feature_dim1, num_points1)
        l1_points = self.IG1.fuse(x1, l1_points)

        # Level 2
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l2_points = self.IG2.fuse(x2, l2_points)

        # Level 3
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = self.IG3.fuse(x3, l3_points)

        # Level 4
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l4_points = self.IG4.fuse(x4, l4_points)

        # Level 5
        l5_xyz, l5_points = self.sa5(l4_xyz, l4_points)
        l5_points = self.IG5.fuse(x5, l5_points)

        fused_features = [l1_points, l2_points, l3_points, l4_points, l5_points]
        return fused_features
class DCM_judge(nn.Module):
    def __init__(self, n_layers=5, c=1, n=8, padding_list=None, norm='bn', dropout=0.1):
        super(DCM_judge, self).__init__()
        if padding_list is None:
            padding_list = [0] * (n_layers + 2)
        self.middle_channel = 2 ** n_layers * n
        self.dropout = dropout
        self.padding_list = padding_list
        self.n_layers = n_layers

        # Down sampling layers
        self.convd_list = nn.ModuleList([
            ConvD(c if i == 0 else 2 ** (i - 1) * n, 2 ** i * n, self.dropout, norm,
                  first=(i == 0), padding=self.padding_list[i])
            for i in range(n_layers + 1)
        ])
        # Up sampling layers
        self.convu_list = nn.ModuleList([
            ConvU(2 ** (i + 1) * n, self.dropout, norm,
                  first=(i == n_layers - 1), padding=self.padding_list[i])
            for i in reversed(range(n_layers))
        ])

        self.pc_evolve = PCModel_wr_v4()

        # Initialize the weights for both sets of layers
        self._initialize_weights()

        # FeatureAggregator and projection heads initialization
        channel_dims = [8, 16, 32, 64, 128]  # Channels at each level
        embedding_dim = 128  # Dimension of the embedding
        self.feature_aggregator = FeatureAggregator(channel_dims, embedding_dim)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self, path):
        # Load the model
        saved_state_dict = torch.load(path, map_location='cpu')

        # Prepare the current model's state dictionary
        model_state_dict = self.state_dict()

        # Track which parameters are loaded
        loaded_params = set()

        # Filter out unnecessary keys from the loaded state dictionary
        for name in list(saved_state_dict.keys()):  # Use list to copy keys
            if name in model_state_dict:
                if model_state_dict[name].size() == saved_state_dict[name].size():
                    loaded_params.add(name)
                else:
                    print(f"Skipping {name} due to size mismatch: Model expects {model_state_dict[name].size()}, "
                          f"loaded state provides {saved_state_dict[name].size()}.")
                    saved_state_dict.pop(name)
            else:
                pass  # Key not in current model's state_dict

        # Update current model's state dictionary with the filtered loaded state dictionary
        model_state_dict.update(saved_state_dict)

        # Load the updated state dictionary back to the model
        self.load_state_dict(model_state_dict, strict=False)

        return loaded_params
    def process_layers(self, x):
        xs = []
        ys = []

        # Downsampling path
        for i, conv in enumerate(self.convd_list):
            x = conv(x)
            if i != len(self.convd_list) - 1:
                xs.append(x)

        y = x

        # Upsampling path
        for i, convu in enumerate(self.convu_list):
            y = convu(y, xs[::-1][i])
            ys.append(y)  # ys will be a list of tensors

        return ys[::-1]  # Return features from the upsampling path

    def global_pooling(self, features):
        # Apply global max pooling over the 'num_points' dimension
        pooled_features = torch.max(features, dim=2)[0]  # Shape: (batch_size, channels_n)
        return pooled_features

    def forward(self, x, coors_g1, coors_g2):
        batch_size = x.shape[0]

        embeddings_1 = []
        embeddings_2 = []

        for b in range(batch_size):
            # Process the image for this batch
            x_batch = x[b:b+1]  # Shape: (1, c, H, W, D)
            ys = self.process_layers(x_batch)  # List of image features

            # Process coors_g1 for this batch
            coors_gs_1 = coors_g1[b:b+1]  # Shape: (1, N, 3, num_points)
            coors_gs_2 = coors_g2[b:b+1]  # Shape: (1, N, 3, num_points)

            # Generate point cloud features for coors_g1
            f_out_1 = self.pc_evolve(coors_gs_1, ys)  # List of features at each level

            # Apply global pooling to each level
            pooled_features_1 = [self.global_pooling(f) for f in f_out_1]

            # Aggregate features using FeatureAggregator
            embedding_1 = self.feature_aggregator(pooled_features_1)  # Shape: (N, embedding_dim)

            # Normalize embeddings
            embedding_1 = F.normalize(embedding_1, p=2, dim=1)

            # Reshape embedding_1 to (1, N, embedding_dim)
            embedding_1 = embedding_1.unsqueeze(0)

            embeddings_1.append(embedding_1)

            # Generate point cloud features for coors_g2
            f_out_2 = self.pc_evolve(coors_gs_2, ys)  # List of features at each level

            # Apply global pooling to each level
            pooled_features_2 = [self.global_pooling(f) for f in f_out_2]

            # Aggregate features using FeatureAggregator
            embedding_2 = self.feature_aggregator(pooled_features_2)  # Shape: (N, embedding_dim)

            # Normalize embeddings
            embedding_2 = F.normalize(embedding_2, p=2, dim=1)

            # Reshape embedding_2 to (1, N, embedding_dim)
            embedding_2 = embedding_2.unsqueeze(0)

            embeddings_2.append(embedding_2)

        # Concatenate embeddings over the batch dimension
        stacked_embedding_1 = torch.cat(embeddings_1, dim=0)  # Shape: (batch_size, N, embedding_dim)
        stacked_embedding_2 = torch.cat(embeddings_2, dim=0)  # Shape: (batch_size, N, embedding_dim)

        print(stacked_embedding_1.shape, stacked_embedding_2.shape)
        diff = (stacked_embedding_2 - stacked_embedding_1)
        return diff
