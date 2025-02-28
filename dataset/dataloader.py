from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import h5py
import torch
from torch.utils.data import DataLoader

def normalize(volume, method = 'mm'):
    """Normalize the volume
       method: "zs": z-std normalization
               "mm": min max normalization
    
    """
    if method == 'zs':
        mean = np.mean(volume)
        std = np.std(volume)
        volume = (volume - mean)/std
        volume = volume.astype("float32")
        
    elif method =='mm':
    #     min = -1000
    #     max = 400
        min = np.min(volume)
        max = np.max(volume)
    #     volume[volume < min] = min
    #     volume[volume > max] = max
        volume = (volume - min) / (max - min)
        volume = volume.astype("float32")
    elif method =='special':
        ana = volume[volume!=0]
        min = np.min(ana)
        max = np.max(ana)
        volume[volume!=0] = (volume[volume!=0] - min) / (max - min)

        
        
    return volume
def center_crop_3d_with_padding(img, crop_size, pad_value=0):
    """
    Center crops a 3D tensor with padding if necessary.

    Parameters:
    - img (torch.Tensor): Input tensor of shape (C, D, H, W).
    - crop_size (tuple): Desired output size (crop_d, crop_h, crop_w).
    - pad_value (float, optional): Value to use for padding. Default is 0.

    Returns:
    - torch.Tensor: Center-cropped (and possibly padded) tensor of shape (C, crop_d, crop_h, crop_w).
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("Input img must be a torch.Tensor.")
    if img.dim() != 4:
        raise ValueError(f"Expected 4D tensor (C, D, H, W), but got {img.dim()}D tensor.")
    if not isinstance(crop_size, (tuple, list)) or len(crop_size) != 3:
        raise ValueError("crop_size must be a tuple or list of three integers (crop_d, crop_h, crop_w).")
    
    C, D, H, W = img.shape
    crop_d, crop_h, crop_w = crop_size

    # Validate crop sizes
    if crop_d <= 0 or crop_h <= 0 or crop_w <= 0:
        raise ValueError("All elements of crop_size must be positive integers.")
    
    # Similarly, pad Width (W) if necessary
    if W < crop_w:
        total_pad_w = crop_w - W
        pad_before_w = total_pad_w // 2
        pad_after_w = total_pad_w - pad_before_w
        # Create padding tensor for Width
        pad_tensor_w_before = torch.full((C, D, H, pad_before_w), pad_value, dtype=img.dtype, device=img.device)
        pad_tensor_w_after = torch.full((C, D, H, pad_after_w), pad_value, dtype=img.dtype, device=img.device)
        # Concatenate along Width dimension
        img = torch.cat([pad_tensor_w_before, img, pad_tensor_w_after], dim=3)
        W = img.shape[3]  # Update W after padding

    # Now perform center cropping
    # Calculate starting indices for each dimension
    start_d = (D - crop_d) // 2
    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2

    # Perform cropping
    cropped_img = img[:, 
                      start_d:start_d + crop_d,
                      start_h:start_h + crop_h,
                      start_w:start_w + crop_w]

    return cropped_img


def center_crop_3d_with_padding_numpy(img, crop_size=(128,128,128), pad_value=0):
    """
    Center crops a 3D NumPy array with padding if necessary.

    Parameters:
    - img (numpy.ndarray): Input array of shape (D, H, W).
    - crop_size (tuple): Desired output size (crop_d, crop_h, crop_w).
    - pad_value (float, optional): Value to use for padding. Default is 0.

    Returns:
    - numpy.ndarray: Center-cropped (and possibly padded) array of shape (crop_d, crop_h, crop_w).
    """
    # Input Validation
    if not isinstance(img, np.ndarray):
        raise TypeError("Input img must be a numpy.ndarray.")
    if img.ndim != 3:
        raise ValueError(f"Expected 3D array (D, H, W), but got {img.ndim}D array.")
    if not isinstance(crop_size, (tuple, list)) or len(crop_size) != 3:
        raise ValueError("crop_size must be a tuple or list of three integers (crop_d, crop_h, crop_w).")
    
    D, H, W = img.shape
    crop_d, crop_h, crop_w = crop_size
    # Validate crop sizes
    if crop_d <= 0 or crop_h <= 0 or crop_w <= 0:
        raise ValueError("All elements of crop_size must be positive integers.")

    # Initialize padding amounts for each dimension
    pad_d_before = pad_d_after = 0
    pad_h_before = pad_h_after = 0
    pad_w_before = pad_w_after = 0

    # Calculate padding for Depth (D) if necessary
    if D < crop_d:
        total_pad_d = crop_d - D
        pad_d_before = total_pad_d // 2
        pad_d_after = total_pad_d - pad_d_before

    # Calculate padding for Height (H) if necessary
    if H < crop_h:
        total_pad_h = crop_h - H
        pad_h_before = total_pad_h // 2
        pad_h_after = total_pad_h - pad_h_before

    # Calculate padding for Width (W) if necessary
    if W < crop_w:
        total_pad_w = crop_w - W
        pad_w_before = total_pad_w // 2
        pad_w_after = total_pad_w - pad_w_before

    # Apply padding if needed
    if any([pad_d_before, pad_d_after, pad_h_before, pad_h_after, pad_w_before, pad_w_after]):
        # Define padding for each dimension: ((D_before, D_after), (H_before, H_after), (W_before, W_after))
        padding = (
            (pad_d_before, pad_d_after),
            (pad_h_before, pad_h_after),
            (pad_w_before, pad_w_after)
        )
        img = np.pad(img, pad_width=padding, mode='constant', constant_values=pad_value)
        # Update dimensions after padding
        D, H, W = img.shape

    # Calculate starting indices for cropping
    start_d = (D - crop_d) // 2
    start_h = (H - crop_h) // 2
    start_w = (W - crop_w) // 2

    # Perform center cropping
    cropped_img = img[start_d:start_d + crop_d,
                      start_h:start_h + crop_h,
                      start_w:start_w + crop_w]

    return cropped_img


class BrainDataset(Dataset):
    def __init__(self, t1_paths, t1gd_paths, t2_paths, flair_paths, label_paths, 
                 transforms=None, is_train=True, **kwargs):
        """
        Args:
            t1_paths:       List of paths to T1 NIfTI files
            t1gd_paths:     List of paths to T1Gd NIfTI files
            t2_paths:       List of paths to T2 NIfTI files
            flair_paths:    List of paths to FLAIR NIfTI files
            label_paths:    List of paths to label NIfTI files
            transforms:     Data augmentations
            is_train:       Boolean indicating training or testing mode
        """
        self.t1_paths = t1_paths
        self.t1gd_paths = t1gd_paths
        self.t2_paths = t2_paths
        self.flair_paths = flair_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.is_train = is_train

        self.modalities = ['t1', 't1gd', 't2', 'flair']

        # Define the label mapping here
        self.class_mapping = {0: 0, 1: 1, 2: 2, 4: 3}  # Map class 4 to index 3

        self.dataset = []
        self.build_dataset()

    def build_dataset(self):
        for idx in range(len(self.t1_paths)):
            # Load the volumes
            t1_img    = nib.load(self.t1_paths[idx]).get_fdata()
            t1gd_img  = nib.load(self.t1gd_paths[idx]).get_fdata()
            t2_img    = nib.load(self.t2_paths[idx]).get_fdata()
            flair_img = nib.load(self.flair_paths[idx]).get_fdata()
            label_img = nib.load(self.label_paths[idx]).get_fdata()

            # Stack the modalities into one array to find the non-zero region (shape: [4, D, H, W])
            patient_image = np.stack([t1_img, t1gd_img, t2_img, flair_img], axis=0)

            # Identify the bounding box around non-zero voxels
            mask = np.sum(patient_image, axis=0) != 0
            z_idxs, y_idxs, x_idxs = np.nonzero(mask)

            # Compute bounding box with a 1-voxel margin, ensuring indices don't go below 0
            zmin = max(0, np.min(z_idxs) - 1)
            zmax = np.max(z_idxs) + 1
            ymin = max(0, np.min(y_idxs) - 1)
            ymax = np.max(y_idxs) + 1
            xmin = max(0, np.min(x_idxs) - 1)
            xmax = np.max(x_idxs) + 1

            # Crop each modality
            t1_img    = t1_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t1gd_img  = t1gd_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t2_img    = t2_img[zmin:zmax, ymin:ymax, xmin:xmax]
            flair_img = flair_img[zmin:zmax, ymin:ymax, xmin:xmax]
            label_img = label_img[zmin:zmax, ymin:ymax, xmin:xmax]
          
            # Center-crop (with padding if needed)
            t1_img    = center_crop_3d_with_padding_numpy(t1_img)
            t1gd_img  = center_crop_3d_with_padding_numpy(t1gd_img)
            t2_img    = center_crop_3d_with_padding_numpy(t2_img)
            flair_img = center_crop_3d_with_padding_numpy(flair_img)
            label_img = center_crop_3d_with_padding_numpy(label_img)

            data_dict = {
                't1': t1_img,
                't1gd': t1gd_img,
                't2': t2_img,
                'flair': flair_img,
                'label': label_img,
                'patient_idx': idx+1
            }
            self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict = self.dataset[index]

        # Get the images and label
        t1_data = data_dict['t1']
        t1gd_data = data_dict['t1gd']
        t2_data = data_dict['t2']
        flair_data = data_dict['flair']
        label_data = data_dict['label']
        patient_idx = data_dict['patient_idx']
        from utils.CT_sampling_support import (
             convert2diploar, extract_boundary,
            spherical_to_cartesian, shrink_or_expand_points, clip_points_within_radius,r_max_region,fps,
            contour_transition_equally, equally_sampling, get_values_at_coords, get_patches_at_coords,equally_sampling_spherical,calculate_centroid
        )

        # Define parameters
        n_regions_theta = 128  # Number of theta bins
        n_regions_phi = 128    # Number of phi bins
    
        
        theta_edges = np.linspace(0, 2*np.pi, n_regions_theta+1)  # 128 bins for theta
        phi_edges = np.linspace(0,   np.pi, n_regions_phi+1)  # 128 bins for phi
        center_TC = calculate_centroid(label_data)
        coordinates = convert2diploar(label_data)
   

        # Prepare bins for theta and phi

        select_points_TC,select_points_WT,select_points_HT = extract_boundary(t1_data, label_data,coordinates,n_regions_phi, n_regions_theta)
        r_stat_center = np.zeros(shape=(n_regions_theta,n_regions_phi))
        r_stat_TC = r_max_region(select_points_TC)
        r_stat_WT = r_max_region(select_points_WT)
        r_stat_HT = r_max_region(select_points_HT)

        # print('0: TC',select_points_TC.shape,'WT',select_points_WT.shape,'HT',select_points_HT.shape)
        TC_points_xyz = spherical_to_cartesian(select_points_TC) 
        WT_points_xyz = spherical_to_cartesian(select_points_WT)   
        HT_points_xyz = spherical_to_cartesian(select_points_HT)   

        TC_points_xyz = fps(TC_points_xyz,4096)
        WT_points_xyz = fps(WT_points_xyz,4096)
        HT_points_xyz = fps(HT_points_xyz,4096)
        # print('1: TC',TC_points_xyz.shape,'WT',WT_points_xyz.shape,'HT',HT_points_xyz.shape)
        rate = 0.02

        all_TC_s = []
        all_d_tcs = []
        all_TC_l = []
        all_d_tcl = []
        all_WT_s = []
        all_d_wts = []
        all_WT_l = []
        all_d_wtl = []

        # Loop to run the function 4 times
        for _ in range(4):
            TC_s, d_tcs = equally_sampling_spherical(TC_points_xyz, center_TC, r_stat_center, r_stat_TC, -rate, 50, theta_edges, phi_edges, withinTC=True)
            TC_l, d_tcl = equally_sampling_spherical(TC_points_xyz, center_TC, r_stat_TC, r_stat_WT, rate, 50, theta_edges, phi_edges)
            WT_s, d_wts = equally_sampling_spherical(WT_points_xyz, center_TC, r_stat_TC, r_stat_WT, -rate, 50, theta_edges, phi_edges)
            WT_l, d_wtl = equally_sampling_spherical(WT_points_xyz, center_TC, r_stat_WT, r_stat_HT, rate, 50, theta_edges, phi_edges, outWT=True)

            # Append results to lists
            all_TC_s.append(TC_s)
            all_d_tcs.append(d_tcs)
            all_TC_l.append(TC_l)
            all_d_tcl.append(d_tcl)
            all_WT_s.append(WT_s)
            all_d_wts.append(d_wts)
            all_WT_l.append(WT_l)
            all_d_wtl.append(d_wtl)

        # Concatenate results
        final_TC_s = np.concatenate(all_TC_s, axis=0)
        final_d_tcs = np.concatenate(all_d_tcs, axis=0)
        final_TC_l = np.concatenate(all_TC_l, axis=0)
        final_d_tcl = np.concatenate(all_d_tcl, axis=0)
        final_WT_s = np.concatenate(all_WT_s, axis=0)
        final_d_wts = np.concatenate(all_d_wts, axis=0)
        final_WT_l = np.concatenate(all_WT_l, axis=0)
        final_d_wtl = np.concatenate(all_d_wtl, axis=0)

        t1_data = normalize( t1_data )
        t1gd_data = normalize( t1gd_data )
        t2_data = normalize(  t2_data )
        flair_data = normalize( flair_data  )

        
        data = {
            'img': {
                    't1': t1_data,
                    't1gd': t1gd_data,
                    't2': t2_data,
                    'flair_data': flair_data,
                    'label_img' : label_data 
            },

            'cts': {
                'TC_s': (final_TC_s,final_d_tcs),
                'TC_l': (final_TC_l,final_d_tcl),
                'WT_s': (final_WT_s,final_d_wts),
                'WT_l': (final_WT_l,final_d_wtl)
            },
            'bound': {
                        'tc':TC_points_xyz,
                        'wt':WT_points_xyz,
                        'ht':HT_points_xyz
                
            },
            'patient_idx': patient_idx
        }

        return data
    

class BrainDataset_WTonly(Dataset):
    def __init__(self, t1_paths, t1gd_paths, t2_paths, flair_paths, label_paths, 
                 transforms=None, is_train=True, **kwargs):
        """
        Args:
            t1_paths:       List of paths to T1 NIfTI files
            t1gd_paths:     List of paths to T1Gd NIfTI files
            t2_paths:       List of paths to T2 NIfTI files
            flair_paths:    List of paths to FLAIR NIfTI files
            label_paths:    List of paths to label NIfTI files
            transforms:     Data augmentations
            is_train:       Boolean indicating training or testing mode
        """
        self.t1_paths = t1_paths
        self.t1gd_paths = t1gd_paths
        self.t2_paths = t2_paths
        self.flair_paths = flair_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.is_train = is_train

        self.modalities = ['t1', 't1gd', 't2', 'flair']

        # Define the label mapping here
        self.class_mapping = {0: 0, 1: 1, 2: 2, 4: 3}  # Map class 4 to index 3

        self.dataset = []
        self.build_dataset()

    def build_dataset(self):
        for idx in range(len(self.t1_paths)):
            # Load the volumes
            t1_img    = nib.load(self.t1_paths[idx]).get_fdata()
            t1gd_img  = nib.load(self.t1gd_paths[idx]).get_fdata()
            t2_img    = nib.load(self.t2_paths[idx]).get_fdata()
            flair_img = nib.load(self.flair_paths[idx]).get_fdata()
            label_img = nib.load(self.label_paths[idx]).get_fdata()

            # Stack the modalities into one array to find the non-zero region (shape: [4, D, H, W])
            patient_image = np.stack([t1_img, t1gd_img, t2_img, flair_img], axis=0)

            # Identify the bounding box around non-zero voxels
            mask = np.sum(patient_image, axis=0) != 0
            z_idxs, y_idxs, x_idxs = np.nonzero(mask)

            # Compute bounding box with a 1-voxel margin, ensuring indices don't go below 0
            zmin = max(0, np.min(z_idxs) - 1)
            zmax = np.max(z_idxs) + 1
            ymin = max(0, np.min(y_idxs) - 1)
            ymax = np.max(y_idxs) + 1
            xmin = max(0, np.min(x_idxs) - 1)
            xmax = np.max(x_idxs) + 1

            # Crop each modality
            t1_img    = t1_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t1gd_img  = t1gd_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t2_img    = t2_img[zmin:zmax, ymin:ymax, xmin:xmax]
            flair_img = flair_img[zmin:zmax, ymin:ymax, xmin:xmax]
            label_img = label_img[zmin:zmax, ymin:ymax, xmin:xmax]
          
            # Center-crop (with padding if needed)
            t1_img    = center_crop_3d_with_padding_numpy(t1_img)
            t1gd_img  = center_crop_3d_with_padding_numpy(t1gd_img)
            t2_img    = center_crop_3d_with_padding_numpy(t2_img)
            flair_img = center_crop_3d_with_padding_numpy(flair_img)
            label_img = center_crop_3d_with_padding_numpy(label_img)

            data_dict = {
                't1': t1_img,
                't1gd': t1gd_img,
                't2': t2_img,
                'flair': flair_img,
                'label': label_img,
                'patient_idx': idx+1
            }
            self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict = self.dataset[index]

        # Get the images and label
        t1_data = data_dict['t1']
        t1gd_data = data_dict['t1gd']
        t2_data = data_dict['t2']
        flair_data = data_dict['flair']
        label_data = data_dict['label']
        patient_idx = data_dict['patient_idx']
        from utils.CT_sampling_support import (
             convert2diploar, extract_boundary,
            spherical_to_cartesian, shrink_or_expand_points, clip_points_within_radius,r_max_region,fps,
            contour_transition_equally, equally_sampling, get_values_at_coords, get_patches_at_coords,equally_sampling_spherical,calculate_centroid
        )

        # Define parameters
        n_regions_theta = 128  # Number of theta bins
        n_regions_phi = 128    # Number of phi bins
    
        
        theta_edges = np.linspace(0, 2*np.pi, n_regions_theta+1)  # 128 bins for theta
        phi_edges = np.linspace(0,   np.pi, n_regions_phi+1)  # 128 bins for phi
        center_TC = calculate_centroid(label_data)
        coordinates = convert2diploar(label_data)
   

        # Prepare bins for theta and phi

        select_points_TC,select_points_WT,select_points_HT = extract_boundary(t1_data, label_data,coordinates,n_regions_phi, n_regions_theta)
        r_stat_center = np.zeros(shape=(n_regions_theta,n_regions_phi))
        r_stat_TC = r_max_region(select_points_TC)
        r_stat_WT = r_max_region(select_points_WT)
        r_stat_HT = r_max_region(select_points_HT)

        # print('0: TC',select_points_TC.shape,'WT',select_points_WT.shape,'HT',select_points_HT.shape)
        TC_points_xyz = spherical_to_cartesian(select_points_TC) 
        WT_points_xyz = spherical_to_cartesian(select_points_WT)   
        HT_points_xyz = spherical_to_cartesian(select_points_HT)   

        TC_points_xyz = fps(TC_points_xyz,4096)
        WT_points_xyz = fps(WT_points_xyz,4096)
        HT_points_xyz = fps(HT_points_xyz,4096)
        # print('1: TC',TC_points_xyz.shape,'WT',WT_points_xyz.shape,'HT',HT_points_xyz.shape)
        rate = 0.005

        all_TC_s = []
        all_d_tcs = []
        all_TC_l = []
        all_d_tcl = []
        all_WT_s = []
        all_d_wts = []
        all_WT_l = []
        all_d_wtl = []


        # Loop to run the function 4 times
        for _ in range(1):
            
            TC_l, d_tcl = equally_sampling_spherical(TC_points_xyz, center_TC, r_stat_TC, r_stat_WT, rate, 300, theta_edges, phi_edges)
            WT_s, d_wts = equally_sampling_spherical(WT_points_xyz, center_TC, r_stat_TC, r_stat_WT, -rate, 300, theta_edges, phi_edges)
            all_TC_l.append(TC_l)
            all_d_tcl.append(d_tcl)
            all_WT_s.append(WT_s)
            all_d_wts.append(d_wts)

        rate2 = 0.02
        for _ in range(4):
            TC_s, d_tcs = equally_sampling_spherical(TC_points_xyz, center_TC, r_stat_center, r_stat_TC, -rate2, 50, theta_edges, phi_edges, withinTC=True)
            WT_l, d_wtl = equally_sampling_spherical(WT_points_xyz, center_TC, r_stat_WT, r_stat_HT, rate2, 50, theta_edges, phi_edges, outWT=True)
            # Append results to lists
            all_TC_s.append(TC_s)
            all_d_tcs.append(d_tcs)
            all_WT_l.append(WT_l)
            all_d_wtl.append(d_wtl)
            # Append results to lists
   



        # Concatenate results

        final_TC_s = np.concatenate(all_TC_s, axis=0)
        final_d_tcs = np.concatenate(all_d_tcs, axis=0)
        final_TC_l = np.concatenate(all_TC_l, axis=0)
        final_d_tcl = np.concatenate(all_d_tcl, axis=0)
        final_WT_s = np.concatenate(all_WT_s, axis=0)
        final_d_wts = np.concatenate(all_d_wts, axis=0)
        final_WT_l = np.concatenate(all_WT_l, axis=0)
        final_d_wtl = np.concatenate(all_d_wtl, axis=0)

        data = {
            'img': {
                    't1': t1_data,
                    't1gd': t1gd_data,
                    't2': t2_data,
                    'flair_data': flair_data,
                    'label_img' : label_data 
            },
            'cts': {
                'TC_s': (final_TC_s,final_d_tcs),
                'TC_l': (final_TC_l,final_d_tcl),
                'WT_s': (final_WT_s,final_d_wts),
                'WT_l': (final_WT_l,final_d_wtl)
            },
            'bound': {
                        'tc':TC_points_xyz,
                        'wt':WT_points_xyz,
                        'ht':HT_points_xyz
                
            },
            'center':  center_TC,
            'patient_idx': patient_idx
        }
        return data
    

class BrainDataset_Conly(Dataset):
    def __init__(self, t1_paths, t1gd_paths, t2_paths, flair_paths, label_paths, 
                 transforms=None, is_train=True, **kwargs):
        """
        Args:
            t1_paths:       List of paths to T1 NIfTI files
            t1gd_paths:     List of paths to T1Gd NIfTI files
            t2_paths:       List of paths to T2 NIfTI files
            flair_paths:    List of paths to FLAIR NIfTI files
            label_paths:    List of paths to label NIfTI files
            transforms:     Data augmentations
            is_train:       Boolean indicating training or testing mode
        """
        self.t1_paths = t1_paths
        self.t1gd_paths = t1gd_paths
        self.t2_paths = t2_paths
        self.flair_paths = flair_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.is_train = is_train

        self.modalities = ['t1', 't1gd', 't2', 'flair']

        # Define the label mapping here
        self.class_mapping = {0: 0, 1: 1, 2: 2, 4: 3}  # Map class 4 to index 3

        self.dataset = []
        self.build_dataset()

    def build_dataset(self):
        for idx in range(len(self.t1_paths)):
            # Load the volumes
            t1_img    = nib.load(self.t1_paths[idx]).get_fdata()
            t1gd_img  = nib.load(self.t1gd_paths[idx]).get_fdata()
            t2_img    = nib.load(self.t2_paths[idx]).get_fdata()
            flair_img = nib.load(self.flair_paths[idx]).get_fdata()
            label_img = nib.load(self.label_paths[idx]).get_fdata()

            # Stack the modalities into one array to find the non-zero region (shape: [4, D, H, W])
            patient_image = np.stack([t1_img, t1gd_img, t2_img, flair_img], axis=0)

            # Identify the bounding box around non-zero voxels
            mask = np.sum(patient_image, axis=0) != 0
            z_idxs, y_idxs, x_idxs = np.nonzero(mask)

            # Compute bounding box with a 1-voxel margin, ensuring indices don't go below 0
            zmin = max(0, np.min(z_idxs) - 1)
            zmax = np.max(z_idxs) + 1
            ymin = max(0, np.min(y_idxs) - 1)
            ymax = np.max(y_idxs) + 1
            xmin = max(0, np.min(x_idxs) - 1)
            xmax = np.max(x_idxs) + 1

            # Crop each modality
            t1_img    = t1_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t1gd_img  = t1gd_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t2_img    = t2_img[zmin:zmax, ymin:ymax, xmin:xmax]
            flair_img = flair_img[zmin:zmax, ymin:ymax, xmin:xmax]
            label_img = label_img[zmin:zmax, ymin:ymax, xmin:xmax]
          
            # Center-crop (with padding if needed)
            t1_img    = center_crop_3d_with_padding_numpy(t1_img)
            t1gd_img  = center_crop_3d_with_padding_numpy(t1gd_img)
            t2_img    = center_crop_3d_with_padding_numpy(t2_img)
            flair_img = center_crop_3d_with_padding_numpy(flair_img)
            label_img = center_crop_3d_with_padding_numpy(label_img)

            data_dict = {
                't1': t1_img,
                'label': label_img,
                'patient_idx': idx+1
            }
            self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict = self.dataset[index]

        # Get the images and label
        t1_data = data_dict['t1']
        label_data = data_dict['label']
        patient_idx = data_dict['patient_idx']
        from utils.CT_sampling_support import     calculate_centroid



        # Define parameters
        n_regions_theta = 128  # Number of theta bins
        n_regions_phi = 128    # Number of phi bins
    
        
        theta_edges = np.linspace(0, 2*np.pi, n_regions_theta+1)  # 128 bins for theta
        phi_edges = np.linspace(0,   np.pi, n_regions_phi+1)  # 128 bins for phi
        center_TC = calculate_centroid(label_data)
        
        data = {
            'center':  center_TC,
            'patient_idx': patient_idx
        }
        return data

class BrainDataset_light(Dataset):
    def __init__(self, t1_paths, t1gd_paths, t2_paths, flair_paths, label_paths, 
                 transforms=None, is_train=True, **kwargs):
        """
        Args:
            t1_paths:       List of paths to T1 NIfTI files
            t1gd_paths:     List of paths to T1Gd NIfTI files
            t2_paths:       List of paths to T2 NIfTI files
            flair_paths:    List of paths to FLAIR NIfTI files
            label_paths:    List of paths to label NIfTI files
            transforms:     Data augmentations
            is_train:       Boolean indicating training or testing mode
        """
        self.t1_paths = t1_paths
        self.t1gd_paths = t1gd_paths
        self.t2_paths = t2_paths
        self.flair_paths = flair_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.is_train = is_train

        self.modalities = ['t1', 't1gd', 't2', 'flair']

        # Define the label mapping here
        self.class_mapping = {0: 0, 1: 1, 2: 2, 4: 3}  # Map class 4 to index 3

        self.dataset = []
        self.build_dataset()

    def build_dataset(self):
        for idx in range(len(self.t1_paths)):
            # Load the volumes
            t1_img    = nib.load(self.t1_paths[idx]).get_fdata()
            t1gd_img  = nib.load(self.t1gd_paths[idx]).get_fdata()
            t2_img    = nib.load(self.t2_paths[idx]).get_fdata()
            flair_img = nib.load(self.flair_paths[idx]).get_fdata()
            label_img = nib.load(self.label_paths[idx]).get_fdata()

            # Stack the modalities into one array to find the non-zero region (shape: [4, D, H, W])
            patient_image = np.stack([t1_img, t1gd_img, t2_img, flair_img], axis=0)

            # Identify the bounding box around non-zero voxels
            mask = np.sum(patient_image, axis=0) != 0
            z_idxs, y_idxs, x_idxs = np.nonzero(mask)

            # Compute bounding box with a 1-voxel margin, ensuring indices don't go below 0
            zmin = max(0, np.min(z_idxs) - 1)
            zmax = np.max(z_idxs) + 1
            ymin = max(0, np.min(y_idxs) - 1)
            ymax = np.max(y_idxs) + 1
            xmin = max(0, np.min(x_idxs) - 1)
            xmax = np.max(x_idxs) + 1

            # Crop each modality
            t1_img    = t1_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t1gd_img  = t1gd_img[zmin:zmax, ymin:ymax, xmin:xmax]
            t2_img    = t2_img[zmin:zmax, ymin:ymax, xmin:xmax]
            flair_img = flair_img[zmin:zmax, ymin:ymax, xmin:xmax]
            label_img = label_img[zmin:zmax, ymin:ymax, xmin:xmax]
          
            # Center-crop (with padding if needed)
            t1_img    = center_crop_3d_with_padding_numpy(t1_img)
            t1gd_img  = center_crop_3d_with_padding_numpy(t1gd_img)
            t2_img    = center_crop_3d_with_padding_numpy(t2_img)
            flair_img = center_crop_3d_with_padding_numpy(flair_img)
            label_img = center_crop_3d_with_padding_numpy(label_img)

            data_dict = {
                't1': t1_img,
                't1gd': t1gd_img,
                't2': t2_img,
                'flair': flair_img,
                'label': label_img,
                'patient_idx': idx+1
            }
            self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_dict = self.dataset[index]

        # Get the images and label
        t1_data = data_dict['t1']
        t1gd_data = data_dict['t1gd']
        t2_data = data_dict['t2']
        flair_data = data_dict['flair']
        label_data = data_dict['label']
       


        t1_data = normalize( t1_data )
        t1gd_data = normalize( t1gd_data )
        t2_data = normalize(  t2_data )
        flair_data = normalize( flair_data  )

        
        data = {
            'img': {
                    't1': t1_data,
                    't1gd': t1gd_data,
                    't2': t2_data,
                    'flair_data': flair_data,
                    'label_img' : label_data 
            }
        }

        return data
    
    
    

def load_from_hdf5(group, keys_to_load=None):
    result = {}
    # Load attributes at the current level
    for attr_key in group.attrs:
        result[attr_key] = group.attrs[attr_key]
    # Determine keys to load
    if keys_to_load is None:
        keys_to_load = list(group.keys())
    elif isinstance(keys_to_load, dict):
        keys_to_load = keys_to_load.keys()
    # Load datasets and groups
    for key in keys_to_load:
        if key in group:
            item = group[key]
            if isinstance(item, h5py.Dataset):
                result[key] = item[:]
            elif isinstance(item, h5py.Group):
                # If keys_to_load is a dict, get subkeys for this group
                sub_keys_to_load = None
                if isinstance(keys_to_load, dict):
                    sub_keys_to_load = keys_to_load.get(key, None)
                result[key] = load_from_hdf5(item, keys_to_load=sub_keys_to_load)
        else:
            print(f"Key '{key}' not found in HDF5 group.")
    return result
class HDF5BrainDataset(Dataset):
    def __init__(self, h5_data_dir):
        self.h5_data_dir = h5_data_dir
#         self.data_files = sorted([f for f in os.listdir(h5_data_dir) if f.endswith('.h5')])
        self.data_files = sorted(
            [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')],
            key=lambda x: int(os.path.splitext(x)[0].split('_')[1])
        )
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        h5_file_name = self.data_files[index]
        h5_file_path = os.path.join(self.h5_data_dir, h5_file_name)
  

        with h5py.File(h5_file_path, 'r') as hf:
            # Specify the keys to load
            keys_to_load = {
                'img': None,    # Load all datasets under 'img'
                'cts': None,    # Load all datasets under 'cts'
                # 'bound' is not included, so it will be skipped
            }
            data = load_from_hdf5(hf, keys_to_load=keys_to_load)

        return data

class HDF5BrainDataset_light(Dataset):
    def __init__(self, h5_data_dir):
        self.h5_data_dir = h5_data_dir
        self.data_files = sorted(
            [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')],
            key=lambda x: int(os.path.splitext(x)[0].split('_')[1])
        )

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        h5_file_name = self.data_files[index]
        h5_file_path = os.path.join(self.h5_data_dir, h5_file_name)
  
        with h5py.File(h5_file_path, 'r') as hf:
            # For the top-level we load:
            #   - all datasets under 'img'
            #   - for 'cts', we load every subgroup but only load 'item_1' and 'item_2' in each subgroup.
            keys_to_load = {
                'cts': {
                    '*': {  # For every subgroup inside 'cts'
                        'item_1': None
                    }
                },
            }
            data = load_from_hdf5(hf, keys_to_load=keys_to_load)

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
class HDF5BrainDataset_v2(Dataset):
    def __init__(self, h5_data_dir):
        self.h5_data_dir = h5_data_dir
        self.data_files = sorted(
            [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')],
            key=lambda x: int(os.path.splitext(x)[0].split('_')[1])
        )

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        h5_file_name = self.data_files[index]
        h5_file_path = os.path.join(self.h5_data_dir, h5_file_name)
  
        with h5py.File(h5_file_path, 'r') as hf:
            # For the top-level we load:
            #   - all datasets under 'img'
            #   - for 'cts', we load every subgroup but only load 'item_1' and 'item_2' in each subgroup.
            keys_to_load = {
                'img_embedding': None,
                'cts': {
                    '*': {  # For every subgroup inside 'cts'
                        'item_1': None,
                        'item_2': None,
                    }
                },
            }
            data = load_from_hdf5(hf, keys_to_load=keys_to_load)

        return data

class HDF5BrainDataset_v3(Dataset):
    def __init__(self, h5_data_dir):
        self.h5_data_dir = h5_data_dir
        self.data_files = sorted(
            [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')],
            key=lambda x: int(os.path.splitext(x)[0].split('_')[1])
        )
        self.dataset = []  # This will store all loaded samples.
        self.build_dataset()

    def build_dataset(self):
        for fname in self.data_files:
            h5_file_path = os.path.join(self.h5_data_dir, fname)
            with h5py.File(h5_file_path, 'r') as hf:
                keys_to_load = {
                    'img_embedding': None,
                    'cts': {
                        '*': {  # For every subgroup inside 'cts'
                            'item_1': None,
                            'item_2': None,
                        }
                    },
                }
                data = load_from_hdf5(hf, keys_to_load=keys_to_load)
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]



# class HDF5BrainDataset_Preprocessed(Dataset):
#     def __init__(self, h5_data_dir):
#         self.h5_data_dir = h5_data_dir
#         self.data_files = sorted(
#             [f for f in os.listdir(h5_data_dir) if f.endswith('.h5')],
#             key=lambda x: int(os.path.splitext(x)[0].split('_')[1])
#         )
#         self.dataset = []  # Preload all samples into memory.
#         self.build_dataset()

#     def build_dataset(self):
#         for fname in self.data_files:
#             h5_path = os.path.join(self.h5_data_dir, fname)
#             with h5py.File(h5_path, 'r') as hf:
#                 # Directly load the preprocessed arrays.
#                 img_emb = hf['img_embedding'][...]
#                 all_g_embs = hf['contour_embedding'][...]
#                 all_scores = hf['contour_scores'][...]
#             # Append the loaded data to our dataset list.
#             self.dataset.append({
#                 'img_embedding': img_emb,
#                 'contour_embedding': all_g_embs,
#                 'contour_scores': all_scores,
#             })

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         # Return the preloaded sample.
#         return self.dataset[index]
