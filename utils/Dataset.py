import numpy as np

from torch.utils.data import Dataset
import os
import h5py
import torch


def calculate_centroid(label, label_value):
    indices = np.argwhere(label == label_value)
    centroid = np.mean(indices, axis=0)
    return centroid
def convert2diploar(label):

    center_TC = calculate_centroid(label, 1)
    center_x, center_y, center_z = center_TC

    # Generate the y, x, z coordinate grids (correct axis assignment)
    x_indices, y_indices, z_indices = np.indices((240, 240, 155))

    # Compute the differences relative to the center
    dx = x_indices - center_x  # X-axis differences
    dy = y_indices - center_y  # Y-axis differences
    dz = z_indices - center_z  # Z-axis differences
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    theta = np.arccos(dz / r)
    phi = np.arctan2(dy, dx)

    out = np.stack(( r, theta, phi), axis=-1)
    return out

def extract_coords(mask,r_coords, theta_coords, phi_coords):
    return {
        'r': r_coords[mask],
        'theta': theta_coords[mask],
        'phi': phi_coords[mask]
    }
def select_points(coords, priority_mask, n_regions_theta,n_regions_phi,theta_bins,phi_bins):
    selected_points = []
    for i in range(n_regions_theta):
        for j in range(n_regions_phi):
            region_priority = (
                (theta_bins[i] <= coords[priority_mask]['theta']) & (coords[priority_mask]['theta'] < theta_bins[i+1]) &
                (phi_bins[j] <= coords[priority_mask]['phi']) & (coords[priority_mask]['phi'] < phi_bins[j+1])
            )

            if coords[priority_mask]['r'][region_priority].size:
                r_select = np.argmax(coords[priority_mask]['r'][region_priority])
                selected_points.append((
                    coords[priority_mask]['r'][region_priority][r_select],
                    coords[priority_mask]['theta'][region_priority][r_select],
                    coords[priority_mask]['phi'][region_priority][r_select]
                ))

    return np.vstack(selected_points)

def spherical_to_cartesian(spherical):
    x = spherical[:, 0] * np.sin(spherical[:, 1]) * np.cos(spherical[:, 2])
    y = spherical[:, 0] * np.sin(spherical[:, 1]) * np.sin(spherical[:, 2])
    z = spherical[:, 0] * np.cos(spherical[:, 1])
    return np.column_stack((x, y, z))
    
# Define the cartesian_to_spherical function
def cartesian_to_spherical_center(xyz, center):
    shifted_xyz = xyz - center
    x, y, z = shifted_xyz[:, 0], shifted_xyz[:, 1], shifted_xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)        # Azimuthal angle (-π to π)
    phi = np.arccos(z / (r + 1e-8)) # Polar angle (0 to π), add epsilon to avoid division by zero
    return r, theta, phi


def fps(points, n_samples,random_ini=False):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    if random_ini ==True:
        selected = np.random.randint(100)
    else:
        selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]



def brain_bound(ed_points, mask_00, center_TC, volume_shape):
    '''return maximum radius vectors for the brain bound, given current ed points. '''
    num_theta_bins = 360  # For 1-degree bins over 360 degrees
    num_phi_bins = 180    # For 1-degree bins over 180 degrees

    theta_bins = np.linspace(-np.pi, np.pi, num_theta_bins + 1)
    phi_bins = np.linspace(0, np.pi, num_phi_bins + 1)

    mask_00_indices = np.where(mask_00)[0]
    mask_00_coords = np.array(np.unravel_index(mask_00_indices, volume_shape)).T  # Shape (N, 3)
    # Compute spherical coordinates for mask_00 points
    r_mask00, theta_mask00, phi_mask00 = cartesian_to_spherical_center(mask_00_coords, center_TC)

    # Assign mask_00 points to bins
    theta_indices_mask00 = np.digitize(theta_mask00, theta_bins) - 1
    phi_indices_mask00 = np.digitize(phi_mask00, phi_bins) - 1

    # Ensure indices are within valid range
    theta_indices_mask00 = np.clip(theta_indices_mask00, 0, num_theta_bins - 1)
    phi_indices_mask00 = np.clip(phi_indices_mask00, 0, num_phi_bins - 1)

    # Initialize a 2D array to hold r_max values for each bin
    r_max_bins = np.zeros((num_theta_bins, num_phi_bins))

    # For each bin, find the maximum r among mask_00 points in that bin
    for idx in range(len(r_mask00)):
        t_idx = theta_indices_mask00[idx]
        p_idx = phi_indices_mask00[idx]
        r = r_mask00[idx]
        # Update the maximum r in the bin if current r is larger
        if r > r_max_bins[t_idx, p_idx]:
            r_max_bins[t_idx, p_idx] = r

    r_out124, theta_out124, phi_out124 = cartesian_to_spherical_center(ed_points, center_TC)

    # Assign out124 points to bins
    theta_indices_out124 = np.digitize(theta_out124, theta_bins) - 1
    phi_indices_out124 = np.digitize(phi_out124, phi_bins) - 1

    # Ensure indices are within valid range
    theta_indices_out124 = np.clip(theta_indices_out124, 0, num_theta_bins - 1)
    phi_indices_out124 = np.clip(phi_indices_out124, 0, num_phi_bins - 1)

    # Retrieve r_max for each point in out124 from r_max_bins
    r_max_brain = r_max_bins[theta_indices_out124, phi_indices_out124]
    return r_max_brain


def boundary_points(coordinates, label_data, t1_data,center_TC, n_regions_theta = 128 , n_regions_phi = 128, fps_points = 4096 ):

    # Flatten the combined_array and label_data
    flat_combined = coordinates.reshape(-1, 3)
    flat_labels = label_data.flatten()

    r_coords = flat_combined[:, 0]
    theta_coords = flat_combined[:, 1]
    phi_coords = flat_combined[:, 2]

    # Prepare bins for theta and phi
    theta_bins = np.linspace(0, np.pi, n_regions_theta + 1)
    phi_bins = np.linspace(-np.pi, np.pi, n_regions_phi + 1)


    # Masks creation and combinations
    t1_data_flat = t1_data.flatten()
    mask_brain = t1_data_flat != 0
    flat_labels = flat_labels.flatten()  # Assuming flat_labels needs to be flattened

    mask_0 = flat_labels == 0  # HT
    mask_1 = flat_labels == 1  # TC
    mask_2 = flat_labels == 2  # ED 
    mask_4 = flat_labels == 4  # ED
    mask_00 = np.logical_and(mask_0, mask_brain)
    mask_00 = mask_00 | mask_1 | mask_2 | mask_4  # Combined masks for any condition

    mask_24 = np.logical_or(mask_2, mask_4)
    mask_124 = mask_1 | mask_2 | mask_4  # Combined masks for any condition

    # Extract coordinates based on masks

    coords_1 = extract_coords(mask_1,r_coords, theta_coords, phi_coords)
    coords_24 = extract_coords(mask_24,r_coords, theta_coords, phi_coords)
    coords_00 = extract_coords(mask_00,r_coords, theta_coords, phi_coords)
    coords_124 = extract_coords(mask_124,r_coords, theta_coords, phi_coords)

    # Use the helper function to generate the selected points arrays
    select_points_1 = select_points(coords={'1': coords_1}, priority_mask='1',n_regions_theta = n_regions_theta,n_regions_phi=n_regions_phi,theta_bins=theta_bins, phi_bins=phi_bins)
    select_points_124 = select_points(coords={'124': coords_124}, priority_mask='124',n_regions_theta = n_regions_theta, n_regions_phi=n_regions_phi,theta_bins=theta_bins, phi_bins=phi_bins)

    tc_points_xyz = spherical_to_cartesian(select_points_1) + center_TC
    ed_points_xyz = spherical_to_cartesian(select_points_124) + center_TC

    out_tc = fps(tc_points_xyz,fps_points)
    out_ed = fps(ed_points_xyz,fps_points)

    return  out_tc, out_ed, mask_00


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
