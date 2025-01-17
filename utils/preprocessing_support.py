from torch.utils.data import Dataset
import os
import torch
import h5py
import numpy as np
import tqdm
import torch


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

def save_dict_to_hdf5(data_dict, h5_file_path):
    """
    Writes a (sub)set of data from a Python dictionary to an HDF5 file.
    Overwrites h5_file_path if it already exists.
    """
    # Utility function for writing sub-dicts recursively
    def _write_group(group, data):
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                _write_group(subgroup, value)
            elif isinstance(value, np.ndarray):
                # Save NumPy arrays as datasets, with compression
                group.create_dataset(key, data=value,
                                     compression='gzip', compression_opts=9)
            else:
                # Save simple data types (e.g., numbers, strings) as attributes
                group.attrs[key] = value

    # Create/overwrite the target .h5 file
    with h5py.File(h5_file_path, 'w') as hf:
        _write_group(hf, data_dict)

# Initialize the dataset