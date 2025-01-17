import argparse
import os
import random
import numpy as np


import os
import h5py
import torch
from torch.utils.data import Dataset
import tqdm



from utils.Dataset import  HDF5BrainDataset
from models.DCM import process_labels_and_distances,generate_indices,extract_pairs,contrastive_loss_vectorized


# Step 1: Set up argparse
parser = argparse.ArgumentParser(description="Hyperparameters for the Brain Tumor Segmentation Model")

parser.add_argument("--GPU_id", type=str, default="-1", help="ID for GPUs")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--para", type=bool, default=False, help="name add")


args = parser.parse_args()

# Step 2: Set random seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

GPU_id = args.GPU_id
if GPU_id !='-1':
    print('using GPU: {}'.format(GPU_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_id

    device_ids = list(np.arange(len(GPU_id)//2+1))
    device_ids = [int(device_ids[i]) for i in device_ids]




para = args.para


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


from models.Point_NN import Point_NN


PN_model = Point_NN(
    input_points=4096, 
    num_stages=4, 
    embed_dim=72, 
    k_neighbors=30, 
    beta=1000, 
    alpha=100
).to(device)

PN_model.eval()
print()

if para and device_ids:
    print('ids', device_ids)
    PN_model = torch.nn.DataParallel(PN_model, device_ids=device_ids)
if cuda:
    PN_model.cuda()

def load_from_hdf5(group):
    """Recursively load all data from an HDF5 group into a Python dict."""
    result = {}
    # Load attributes at the current level
    for attr_key in group.attrs:
        result[attr_key] = group.attrs[attr_key]

    # For each key in this group
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            result[key] = item[:]
        elif isinstance(item, h5py.Group):
            # Recursively load sub-group
            result[key] = load_from_hdf5(item)
    return result


class HDF5BrainDataset(Dataset):
    def __init__(self, h5_data_dir):
        self.h5_data_dir = h5_data_dir
        self.data_files = sorted(
            [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')],
            key=lambda x: int(os.path.splitext(x)[0].split('_')[1])
        )
        
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        """
        Reads from HDF5 and returns a Python dictionary with the data. 
        This is READ-ONLY: changes to the returned dict do not save back.
        """
        h5_file_name = self.data_files[index]
        h5_file_path = os.path.join(self.h5_data_dir, h5_file_name)

        with h5py.File(h5_file_path, 'r') as hf:
            # Adjust the keys you actually want to load
            data = load_from_hdf5(hf)
        return data

    def save_item_2(self, index, key_path, embedding_array):
        """
        Opens the HDF5 file in read-write mode and saves `embedding_array`
        under cts/<k_>/item_2. 
        """
        h5_file_name = self.data_files[index]
        h5_file_path = os.path.join(self.h5_data_dir, h5_file_name)

        with h5py.File(h5_file_path, 'r+') as hf:
            # Navigate to, for example, cts/TC_s
            group = hf
            for part in key_path.split('/'):
                group = group[part]

            # If 'item_2' already exists, delete it
            if 'item_2' in group:
                del group['item_2']

            # Create new dataset with compression
            group.create_dataset('item_2', data=embedding_array,
                                 compression='gzip', compression_opts=9)


# Initialize the dataset
h5_data_dir = 'preprocessed_data_BraTS20_h5'
dataset = HDF5BrainDataset(h5_data_dir)

for i in tqdm.tqdm(range(len(dataset))):
    data_dict = dataset[i]  # loads everything from .h5 into memory

    # data_dict['cts'] has keys like 'TC_s', 'TC_l', 'WT_s', 'WT_l'
    for k_ in ['TC_s', 'TC_l', 'WT_s', 'WT_l']:
        # data_dict['cts'][k_] is presumably a dictionary 
        # containing 'item_0', 'item_1', etc.


        x = torch.from_numpy(data_dict['cts'][k_]['item_0']).permute(0,2,1).float()

        with torch.no_grad():
            point_embedding = PN_model(x)  # => shape e.g. [batch_size, embed_dim]

        # Convert to numpy
        embedding_array = point_embedding.cpu().numpy()

        # Save to the .h5 file under cts/<k_>/item_2
        key_path = f'cts/{k_}'
        dataset.save_item_2(i, key_path, embedding_array)
