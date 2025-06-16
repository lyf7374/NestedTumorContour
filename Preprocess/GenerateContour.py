import argparse
import os
import torch
import random
import numpy as np
import tqdm
import torch.nn.functional as F
from models.DCMModel import ContourEncoder, get_image_and_contours_rank
from utils.ContourSupport import process_patient_images, cartesian_to_spherical, generate_samples
from models.Point_NN import Point_NN
from dataset.dataloader import HDF5BrainDataset_v3, HDF5BrainDataset_loadall
from torch.utils.data import Subset, DataLoader

parser = argparse.ArgumentParser(description="Nested Sampling for Brain Tumor Contour Optimization")
parser.add_argument("--GPU_id", type=str, default="-1", help="GPU ID to use; '-1' for CPU")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--para", type=bool, default=False, help="name add")
parser.add_argument("--max_iterations", type=int, default=5000, help="Max iterations for nested sampling")
parser.add_argument("--population_size", type=int, default=10, help="Population size for nested sampling")
parser.add_argument("--factor", type=float, default=0.01, help="Max iterations for nested sampling")
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

factor = args.factor
set_seed(args.seed)
GPU_id = args.GPU_id
if GPU_id != '-1':
    print('using GPU: {}'.format(GPU_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id
    device_ids = list(np.arange(len(GPU_id)//2+1))
    device_ids = [int(device_ids[i]) for i in device_ids]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

max_iterations = args.max_iterations
population_size = args.population_size

model_path = 'saved_models/CMrank_11_inf1_1_recur.pth'
week0_folder = 'recur_data/procesed_perfect_week0'
weekn_folder = 'recur_data/procesed_perfect_week_N'
h5_data_dir = 'perfect_h5'
# Initialize and load the model
model = ContourEncoder(
    img_channels=1024,
    contour_dim=1152,
    hidden_dim=256,
    n_heads=4,
    num_decoder_layers=1
)
state_dict = torch.load(model_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.to(device)

# Use DataParallel if applicable, then ensure model is on GPU
if 'device_ids' in globals() and device_ids:
    print('Using DataParallel with device_ids:', device_ids)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
if cuda:
    model = model.cuda()
model.eval()

PN_model = Point_NN(
    input_points=4096,
    num_stages=4,
    embed_dim=72,
    k_neighbors=30,
    beta=1000,
    alpha=100
).to(device)
PN_model.eval()

dataset = HDF5BrainDataset_loadall(h5_data_dir)
# For demonstration, we split the dataset deterministically:
train_subset = Subset(dataset, list(range(22)))
val_subset = Subset(dataset, list(range(22, 34)))

# val_loader = DataLoader(val_subset, batch_size=1)

val_loader = DataLoader(dataset, batch_size=1)

idx = 20  # sample patient index (adjust as needed)

week0_paths = []
weekn_paths = []
for patient in os.listdir(week0_folder):
    if patient.startswith('.'):
        continue
    week0_paths.append(os.path.join(week0_folder, patient))
    weekn_paths.append(os.path.join(weekn_folder, patient))
week0_paths = sorted(week0_paths, key=lambda x: int(os.path.basename(x).split('-')[-1]))
weekn_paths = sorted(weekn_paths, key=lambda x: int(os.path.basename(x).split('-')[-1]))

week0_data = process_patient_images(week0_paths[idx])
weekn_data = process_patient_images(weekn_paths[idx],
                                    bbox=week0_data['bbox'],
                                    center=week0_data['center'])



for i, val_item in enumerate(val_loader):
    patient_filename = week0_paths[idx].split('/')[-1].replace("-", "_")
    
    # Compare the modified patient filename with the first element in the list
    if patient_filename != val_item['patient_idx'][0]:
        continue
    image_emb, all_g_embs, d_scores = get_image_and_contours_rank(val_item)
    geometric_center = np.array([
        val_item['center']['item_0'].item(),
        val_item['center']['item_1'].item(),
        val_item['center']['item_2'].item()
    ])
    ini_g = val_item['cts']['TC_l']['item_0'][0, -population_size:, :, :]
    image_batch = image_emb.repeat(population_size, 1, 1, 1, 1)
    image_batch_TC = image_emb.repeat(100, 1, 1, 1, 1)
    break
# ----- Move inputs to GPU -----
image_emb = image_emb.to(device)
ini_g = ini_g.to(device)
all_g_embs = all_g_embs.to(device)
image_batch_TC = image_batch_TC.to(device)

print('Patient:', week0_paths[idx], val_item['patient_idx'])
print('Center from week0:', week0_data['center'])
print('Center check:', week0_data['center'], geometric_center)

m1 = week0_data['label']
m2 = weekn_data['label']
tc_comparison = np.zeros_like(m1)
tc_comparison[np.isin(m1, [1, 4]) & np.isin(m2, [1, 4])] = 1  # Overlap
tc_comparison[np.isin(m2, [1, 4]) & ~np.isin(m1, [1, 4])] = 2  # New TC
tc_comparison[m2 == 2] = 3  # m2 Label3

overlap_comparison = np.zeros_like(m1)
new_tc_mask = (tc_comparison == 2)
m1_label2_mask = (m1 == 2)
overlap_comparison[new_tc_mask & m1_label2_mask] = 1
overlap_comparison[new_tc_mask & ~m1_label2_mask] = 2
overlap_comparison[~new_tc_mask & m1_label2_mask] = 3

with torch.no_grad():
  
    g_pc_emb = PN_model(ini_g.permute(0, 2, 1))  # ini_g: (population_size, ...)
    emb_g = model(image_emb.repeat(population_size, 1, 1, 1, 1), g_pc_emb)
    e_tc = model(image_batch_TC, all_g_embs[:100])
    e_tc_norm = F.normalize(e_tc, p=2, dim=-1)
emb_g_norm = F.normalize(emb_g, p=2, dim=-1)
sim_tc_mat = torch.matmul(emb_g_norm, e_tc_norm.transpose(0, 1))
sim_tc_avg = sim_tc_mat.mean(dim=1)

live_points = ini_g.clone()
live_likelihoods = sim_tc_avg.clone()
removed_points = []
removed_likelihoods = []
print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # ')
max_attempts = 100  # limit how many times we try each iteration, to avoid infinite loops

for iteration in tqdm.tqdm(range(max_iterations)):
    # 1. Identify and remove the worst-likelihood sample
    worst_idx = torch.argmin(live_likelihoods)
    worst_point = live_points[worst_idx].unsqueeze(0)
    worst_lh = live_likelihoods[worst_idx].unsqueeze(0)
    removed_points.append(worst_point.cpu())
    removed_likelihoods.append(worst_lh.cpu())

    # Remove it from the live set
    live_points = torch.cat([live_points[:worst_idx], live_points[worst_idx+1:]], dim=0)
    live_likelihoods = torch.cat([live_likelihoods[:worst_idx], live_likelihoods[worst_idx+1:]], dim=0)

    # 2. Keep generating candidate batches until we find at least one
    #    whose likelihood >= worst_lh
    attempts = 0
    new_point = None
    new_lh = None
    factor_in = factor * 1
    while True:
        
        # Sample a new candidate contour from a random parent
        parent_idx = np.random.randint(len(live_points))
        parent = live_points[parent_idx]

        # Convert to spherical coordinates
        geometric_center_tensor = torch.tensor(geometric_center, 
                                               device=parent.device, 
                                               dtype=parent.dtype)
        parent_sph = cartesian_to_spherical((parent - geometric_center_tensor).cpu().numpy())

        # Generate candidate_batch
        candidate_batch = generate_samples(parent_sph, geometric_center, num_samples=20, factor=factor_in)
        candidate_batch = candidate_batch.to(device)

        # Evaluate their likelihood
        with torch.no_grad():
            emb_g_candidate = PN_model(candidate_batch.permute(0, 2, 1))
            emb_candidate_norm = F.normalize(
                model(image_emb.repeat(len(candidate_batch), 1, 1, 1, 1), emb_g_candidate),
                p=2, dim=-1
            )
        sim_mat_candidates = torch.matmul(emb_candidate_norm, e_tc_norm.transpose(0, 1))
        sim_avg_candidates = sim_mat_candidates.mean(dim=1)

        # Keep only those >= worst_lh
        mask = sim_avg_candidates >= worst_lh.item()
        valid = candidate_batch[mask]
        valid_l = sim_avg_candidates[mask]

        if len(valid) > 0:
            # Pick the best valid point
            best_new_idx = torch.argmax(valid_l)
            new_point = valid[best_new_idx].unsqueeze(0)
            new_lh = valid_l[best_new_idx].unsqueeze(0)
            break  # we found a valid replacement

        attempts += 1
        if (attempts+1) % 25 ==0:
            factor_in = factor * 1.5
        if attempts > max_attempts:
            print(f"Warning: Could not find a valid point after {max_attempts} attempts.")
            print("This indicates the region might be too constrained or sampling is insufficient.")
            # As a fallback, just pick the best among all (even if it's worse)
            best_new_idx = torch.argmax(sim_avg_candidates)
            new_point = candidate_batch[best_new_idx].unsqueeze(0)
            new_lh = sim_avg_candidates[best_new_idx].unsqueeze(0)
            break

    # 3. Insert the new point into the live set
    live_points = torch.cat([live_points, new_point], dim=0)
    live_likelihoods = torch.cat([live_likelihoods, new_lh], dim=0)

    # Optional: log progress
    if iteration % 50 == 0:
        print('current_max', torch.max(live_likelihoods))
        print('likelihood', [f"{t.item():.4f}" for t in live_likelihoods])
        print()
    # Early stopping criterion
    if live_likelihoods.mean() > 0.95:
        print("Converged early at iteration", iteration)
        break

print("Nested sampling optimization complete.")
# ================================
# Save relevant results for analysis
# ================================

# Gather removed points and live points into NumPy arrays
all_removed_pts = torch.cat(removed_points, dim=0).cpu().numpy()    # shape: (num_removed, 4096, 3)
all_removed_lhs = torch.cat(removed_likelihoods, dim=0).cpu().numpy()  # shape: (num_removed,)
final_live_pts  = live_points.cpu().numpy()                         # shape: (N_live, 4096, 3)
final_live_lhs  = live_likelihoods.cpu().numpy()                     # shape: (N_live,)

# Combine them into one big set
all_pts = np.concatenate([all_removed_pts, final_live_pts], axis=0)   # shape: (M, 4096, 3)
all_lhs = np.concatenate([all_removed_lhs, final_live_lhs], axis=0)     # shape: (M,)

# Save arrays to disk so you can download them for further analysis
np.save("all_pts.npy", all_pts)
np.save("all_lhs.npy", all_lhs)

