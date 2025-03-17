import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from dataset.dataloader import (
    center_crop_3d_with_padding_numpy_with_center,
    calculate_centroid
)
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes, binary_closing, generate_binary_structure


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_patient_images(path, bbox=None, center=None):
    """
    Load and process patient images from a given directory.
    
    Parameters:
        path (str): Directory with registered images and mask.
        bbox (tuple, optional): Bounding box (zmin, zmax, ymin, ymax, xmin, xmax).
                                If None, computed from the modalities.
        center (array-like, optional): Center for cropping. If None, computed from label mask.
    
    Returns:
        dict: Processed images and metadata.
    """
    t1_img    = nib.load(os.path.join(path, 'registered_T1.nii.gz')).get_fdata()
    t1gd_img  = nib.load(os.path.join(path, 'registered_CT1.nii.gz')).get_fdata()
    t2_img    = nib.load(os.path.join(path, 'registered_T2.nii.gz')).get_fdata()
    flair_img = nib.load(os.path.join(path, 'registered_FLAIR.nii.gz')).get_fdata()
    label_img = nib.load(os.path.join(path, 'registered_T2_mask.nii.gz')).get_fdata()

    if bbox is None:
        patient_image = np.stack([t1_img, t1gd_img, t2_img, flair_img], axis=0)
        mask = np.sum(patient_image, axis=0) != 0
        z_idxs, y_idxs, x_idxs = np.nonzero(mask)
        zmin = max(0, np.min(z_idxs) - 1)
        zmax = np.max(z_idxs) + 1
        ymin = max(0, np.min(y_idxs) - 1)
        ymax = np.max(y_idxs) + 1
        xmin = max(0, np.min(x_idxs) - 1)
        xmax = np.max(x_idxs) + 1
        bbox = (zmin, zmax, ymin, ymax, xmin, xmax)
    else:
        zmin, zmax, ymin, ymax, xmin, xmax = bbox

    t1_img    = t1_img[zmin:zmax, ymin:ymax, xmin:xmax]
    t1gd_img  = t1gd_img[zmin:zmax, ymin:ymax, xmin:xmax]
    t2_img    = t2_img[zmin:zmax, ymin:ymax, xmin:xmax]
    flair_img = flair_img[zmin:zmax, ymin:ymax, xmin:xmax]
    label_img = label_img[zmin:zmax, ymin:ymax, xmin:xmax]

    if center is None:
        center = calculate_centroid(label_img)

    t1_img    = center_crop_3d_with_padding_numpy_with_center(t1_img, center=center)
    t1gd_img  = center_crop_3d_with_padding_numpy_with_center(t1gd_img, center=center)
    t2_img    = center_crop_3d_with_padding_numpy_with_center(t2_img, center=center)
    flair_img = center_crop_3d_with_padding_numpy_with_center(flair_img, center=center)
    label_img = center_crop_3d_with_padding_numpy_with_center(label_img, center=center)

    # Adjust label values: replace label 2 with 4 and label 3 with 2
    label_img[label_img == 2] = 4
    label_img[label_img == 3] = 2

    return {
        't1_img': t1_img,
        't1gd_img': t1gd_img,
        't2_img': t2_img,
        'flair_img': flair_img,
        'label': label_img,
        'bbox': bbox,
        'center': center
    }

def compute_star_shaped_inside_mask(point_cloud, center, volume_shape=(128, 128, 128),
                                    fill_holes=True, closing=True, iterations=1):
    """
    Compute an inside mask for a star-shaped volume defined by a boundary point cloud.
    
    Parameters:
        point_cloud (np.array): (N,3) boundary points.
        center (array-like): A 3D coordinate known to be inside.
        volume_shape (tuple): Shape of the volume.
        fill_holes (bool): Whether to fill holes.
        closing (bool): Whether to apply binary closing.
        iterations (int): Number of closing iterations.
        
    Returns:
        np.array: Binary mask (1=inside, 0=outside).
    """
    grid = np.indices(volume_shape).reshape(3, -1).T
    tree = cKDTree(point_cloud)
    _, nearest_indices = tree.query(grid)
    radial_distances = np.linalg.norm(grid - center, axis=1)
    nearest_pt_radial = np.linalg.norm(point_cloud[nearest_indices] - center, axis=1)
    outside_mask = radial_distances > nearest_pt_radial
    inside_mask = (~outside_mask).reshape(volume_shape).astype(np.uint8)

    if fill_holes or closing:
        structure = generate_binary_structure(rank=3, connectivity=2)
    if fill_holes:
        inside_mask = binary_fill_holes(inside_mask, structure=structure).astype(np.uint8)
    if closing:
        inside_mask = binary_closing(inside_mask, structure=structure, iterations=iterations).astype(np.uint8)
    return inside_mask

def compute_label_volumes(inside_mask, label_array, label_1=1, label_3=3):
    """
    Count voxels with specific labels inside the mask.
    
    Returns:
        tuple: (label_1_count, label_3_count, total_inside)
    """
    inside = (inside_mask == 1)
    label_1_count = np.count_nonzero((label_array == label_1) & inside)
    label_3_count = np.count_nonzero((label_array == label_3) & inside)
    total_inside = np.count_nonzero(inside)
    return label_1_count, label_3_count, total_inside

def cartesian_to_spherical(xyz):
    """
    Convert Cartesian coordinates (N,3) to spherical coordinates (r, theta, phi).
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
    phi = np.arccos(z / (r + 1e-8))
    return np.stack((r, theta, phi), axis=-1)

def spherical_to_cartesian(sph_coords):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian (N,3).
    """
    r, theta, phi = sph_coords[:, 0], sph_coords[:, 1], sph_coords[:, 2]
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack((x, y, z), axis=-1)

def generate_samples(parent_sph, geometric_center, num_samples=10, factor=0.01):
    """
    Generate candidate samples by perturbing the parent contour in spherical coordinates.
    
    Parameters:
        parent_sph (np.array): Parent contour in spherical coordinates.
        geometric_center (np.array): Center to add back after conversion.
        num_samples (int): Number of samples.
        factor (float): Perturbation factor.
    
    Returns:
        torch.Tensor: Samples in Cartesian coordinates.
    """
    samples = np.empty((num_samples, parent_sph.shape[0], parent_sph.shape[1]))
    for i in range(num_samples):
        sample = parent_sph.copy()
        multiplier = 1 - np.sign(factor) * np.abs(
            np.random.normal(0, np.abs(factor), parent_sph.shape[0])
        )
        sample[:, 0] *= multiplier
        samples[i] = spherical_to_cartesian(sample) + geometric_center
    return torch.from_numpy(samples).float()

def evaluate_current_sets(samples, model, image_batch, PN_model):
    """
    Evaluate candidate samples using the provided model.
    
    Parameters:
        samples (torch.Tensor): Candidate contours in Cartesian coordinates.
        model (torch.nn.Module): The evaluation model.
        image_batch (torch.Tensor): Input image batch.
    
    Returns:
        torch.Tensor: Normalized embeddings.
    """
    with torch.no_grad():
        g_pc_emb = PN_model(samples.permute(0, 2, 1))
        emb_g = model(image_batch, g_pc_emb)
    return F.normalize(emb_g, p=2, dim=-1)

