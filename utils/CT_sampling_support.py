import numpy as np
from scipy.stats import binned_statistic_2d
from scipy.ndimage import center_of_mass

# def calculate_centroid(label, label_value):
#     indices = np.argwhere(label == label_value)
#     centroid = np.mean(indices, axis=0)
#     return centroid
# def calculate_centroid(label):
#     indices = np.vstack([np.argwhere(label == 1),np.argwhere((label == 4))])
#     centroid = np.mean(indices, axis=0)
#     return centroid


def calculate_centroid(label):
    """
    Calculate the centroid of regions labeled as 1 or 4.

    Parameters:
    label (ndarray): N-dimensional array with integer labels.

    Returns:
    centroid (tuple or None): Coordinates of the centroid or None if no labels found.
    """
    # Create a binary mask where label is 1 or 4
    mask = np.isin(label, [1, 4])

    # Check if the mask contains any True values
    if not np.any(mask):
        print("No voxels with label 1 or 4 found, using 1,2,4")
        mask = np.isin(label, [0,1,2,4])
        # return None

    # Compute the center of mass
    centroid = center_of_mass(mask)
    return centroid
def convert2diploar(label):
    center_TC = calculate_centroid(label)
    center_x, center_y, center_z = center_TC

    # Generate the x, y, z coordinate grids
    x_indices, y_indices, z_indices = np.indices(label.shape)

    # Compute the differences relative to the center
    dx = x_indices - center_x
    dy = y_indices - center_y
    dz = z_indices - center_z

    r = np.sqrt(dx**2 + dy**2 + dz**2)
    theta = np.arctan2(dy, dx)
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # Adjust theta to range [0, 2π]
    phi = np.arccos(dz / (r+ 1e-8))                     # Polar angle from 0 to π

    out = np.stack((r, theta, phi), axis=-1)
    return out

def fps(points, n_samples, random_ini=False):
    """
    Performs Farthest Point Sampling (FPS) on a point cloud.

    Parameters:
    - points: [N, D] array containing the whole point cloud.
    - n_samples: Number of samples you want in the sampled point cloud.
    - random_ini: If True, starts FPS with a random point.

    Returns:
    - Sampled points of shape [n_samples, D].
    """
    points = np.array(points)
    N = len(points)
    
    # Handle cases where n_samples >= N
    if n_samples >= N:
        # Return all points and pad the rest
        sample_inds = np.arange(N)
        pad_size = n_samples - N
        if pad_size > 0:
            # Option 1: Pad with zeros
            padding = np.zeros((pad_size, points.shape[1]))
            # Option 2: Repeat existing points (uncomment the next line to use this instead)
            # padding = points[np.random.choice(N, pad_size)]
            sampled_points = np.vstack([points[sample_inds], padding])
        else:
            sampled_points = points[sample_inds]
        return sampled_points

    # Initialize variables
    points_left = np.arange(N)  # Indices of points not yet sampled
    sample_inds = np.zeros(n_samples, dtype='int')  # Indices of sampled points
    dists = np.ones(N) * float('inf')  # Initialize distances to infinity

    # Choose the initial point
    if random_ini:
        selected = np.random.randint(N)
    else:
        selected = 0
    sample_inds[0] = selected

    # Remove the selected point from points_left
    points_left = np.delete(points_left, selected)

    # Main loop of FPS
    for i in range(1, n_samples):
        last_added = sample_inds[i - 1]
        dist_to_last_added = np.sum((points[points_left] - points[last_added]) ** 2, axis=1)
        dists[points_left] = np.minimum(dists[points_left], dist_to_last_added)

        # Check if there are still points left to sample
        if len(points_left) == 0:
            # No more points to sample; fill the rest with zeros or repeat points
            pad_size = n_samples - i
            # Option 1: Pad with zeros
            padding_inds = np.full(pad_size, -1)
            # Option 2: Repeat existing points (uncomment the next line to use this instead)
            # padding_inds = np.random.choice(N, pad_size)
            sample_inds[i:] = padding_inds
            break

        # Select the next point that is farthest
        selected_idx = np.argmax(dists[points_left])
        selected = points_left[selected_idx]
        sample_inds[i] = selected

        # Remove the selected point from points_left
        points_left = np.delete(points_left, selected_idx)

    # Gather sampled points
    sampled_points = points[sample_inds[:n_samples]]

    # Handle padding if necessary
    if np.any(sample_inds == -1):
        # Replace -1 indices with zeros (since points[-1] is valid but not intended)
        sampled_points[sample_inds == -1] = 0

    return sampled_points


def extract_boundary(t1_data, label, coordinates, n_regions_phi=128, n_regions_theta=128):
    t1_data_flat = t1_data.flatten()
    mask_brain = t1_data_flat != 0
    
    flat_labels = label.flatten()
    # coordinates = convert2diploar(label)
    flat_combined = coordinates.reshape(-1, 3)
    r_coords = flat_combined[:, 0]
    theta_coords = flat_combined[:, 1]
    phi_coords = flat_combined[:, 2]

    mask_0 = flat_labels == 0  # HT
    mask_1 = flat_labels == 1  # TC
    mask_2 = flat_labels == 2  # ED 
    mask_4 = flat_labels == 4  # ET
    mask_00 = np.logical_and(mask_0, mask_brain)

    mask_24 = np.logical_or(mask_2, mask_4)
    mask_124 = mask_1 | mask_2 | mask_4  # Combined masks for any condition

    mask_WT = mask_124
    mask_TC = mask_1 |  mask_4
    mask_HT = mask_00 | mask_WT
    # mask_HT = mask_00 

    # Function to simplify the extraction of coordinate arrays
    def extract_coords(mask):
        return {
            'r': r_coords[mask],
            'theta': theta_coords[mask],
            'phi': phi_coords[mask]
        }

    def select_points(coords):
        selected_points = []
        for i in range(n_regions_theta):
            for j in range(n_regions_phi):
                region_priority = (
                    (theta_bins[i] <= coords['theta']) & (coords['theta'] < theta_bins[i+1]) &
                    (phi_bins[j] <= coords['phi']) & (coords['phi'] < phi_bins[j+1])
                )

                if coords['r'][region_priority].size:
                    r_select = np.argmax(coords['r'][region_priority])
                    selected_points.append((
                        coords['r'][region_priority][r_select],
                        coords['theta'][region_priority][r_select],
                        coords['phi'][region_priority][r_select]
                    ))

        return np.vstack(selected_points)
    
    coords_TC =  extract_coords(mask_TC) # 14
    coords_WT =  extract_coords(mask_WT) # 14 + 2 
    coords_HT =  extract_coords(mask_HT) # brain and 0 

    # Define regions for selection
    theta_bins = np.linspace(0, 2*np.pi, n_regions_theta + 1)
    phi_bins = np.linspace( 0, np.pi, n_regions_phi + 1)
    # Helper function to perform region checks and point selection

    # Use the helper function to generate the selected points arrays
    select_points_TC = select_points(coords_TC)
    select_points_WT = select_points(coords_WT)
    select_points_HT = select_points(coords_HT)
   
    return select_points_TC,select_points_WT,select_points_HT

def cartesian_to_spherical(xyz):
    shifted_xyz = xyz 
    x, y, z = shifted_xyz[:, 0], shifted_xyz[:, 1], shifted_xyz[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # Adjust theta to range [0, 2π]
    phi = np.arccos(z / (r+ 1e-8))                     # Polar angle from 0 to π

    return np.stack((r, theta, phi), axis=-1)
def spherical_to_cartesian(sph_coords):
    r, theta, phi = sph_coords[:, 0], sph_coords[:, 1], sph_coords[:, 2]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.stack((x, y, z), axis=-1) 



def r_max_region(select_points):
    

    n_regions_theta = 128  # Number of theta bins
    n_regions_phi = 128    # Number of phi bins
    theta_bins = np.linspace(0, 2*np.pi, n_regions_theta + 1)
    phi_bins = np.linspace(0, np.pi, n_regions_phi + 1)  
    
    statistic, _, _, _ = binned_statistic_2d(
        select_points[:,1], select_points[:,2], select_points[:,0], statistic='max', bins=[theta_bins, phi_bins]
    )

    # 'statistic' is a 2D array of shape (128, 128) containing the maximum r in each bin
    # Handle bins with no data (NaN values) if necessary
    statistic = np.nan_to_num(statistic, nan=0.0)
    return statistic



def shrink_or_expand_points(points, center, factor, indices=None):
    """
    Shrink or expand specified points radially around a given center.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) containing the points.
        center (np.ndarray): Array of shape (3,) representing the center point.
        factor (float or np.ndarray): Shrink/expand factor(s).
                                      - If scalar, applies the same factor to all specified points.
                                      - If array of shape (M,), applies per-point factors.
        indices (np.ndarray or None): Indices of points to modify. If None, all points are modified.

    Returns:
        np.ndarray: Transformed points of shape (N, 3).
    """
    if indices is None:
        indices = np.arange(len(points))
    else:
        indices = np.asarray(indices)
    
    # Shift points so center is at origin
    shifted_points = points - center

    # Convert to spherical coordinates
    x, y, z = shifted_points[indices, 0], shifted_points[indices, 1], shifted_points[indices, 2]
    r = np.linalg.norm(shifted_points[indices], axis=1)
    theta = np.arctan2(y, x)        # azimuthal angle
    phi = np.arccos(z / (r + 1e-8)) # polar angle, add small epsilon to avoid division by zero

    # Modify the radius
    if np.isscalar(factor):
        r_new = r * factor
    else:
        factor = np.asarray(factor)
        r_new = r * factor
    # Convert back to Cartesian coordinates
    x_new = r_new * np.sin(phi) * np.cos(theta)
    y_new = r_new * np.sin(phi) * np.sin(theta)
    z_new = r_new * np.cos(phi)

    # Update the points
    transformed_points = points.copy()
    transformed_points[indices] = np.stack((x_new, y_new, z_new), axis=-1) + center
    return transformed_points
import numpy as np

def clip_points_within_radius(points, center, theta_edges,  phi_edges, r_min_map=None, r_max_map=None ):
    """
    Clip points to be within specified radial distances from the center.
    Supports per-point radial bounds based on (theta, phi) bins.

    Parameters:
        points (np.ndarray): Array of shape (N, 3) containing the points.
        center (np.ndarray): Array of shape (3,) representing the center point.
        r_min_map (np.ndarray): 2D array of shape (128, 128) containing r_min values for each (theta, phi) bin.
        r_max_map (np.ndarray): 2D array of shape (128, 128) containing r_max values for each (theta, phi) bin.
        theta_edges (np.ndarray): Array of shape (129,) defining the theta bin edges.
        phi_edges (np.ndarray): Array of shape (129,) defining the phi bin edges.

    Returns:
        np.ndarray: Transformed points of shape (N, 3).
    """
    # Shift points so center is at origin
    shifted_points = points - center

    # Convert to spherical coordinates
    x, y, z = shifted_points[:, 0], shifted_points[:, 1], shifted_points[:, 2]
    r = np.linalg.norm(shifted_points, axis=1)
    
    # Corrected spherical coordinate conversion
    theta = np.arccos(z / (r + 1e-8))  # Polar angle θ ∈ [0, π]
    phi = np.arctan2(y, x)             # Azimuthal angle φ ∈ (-π, π]
    phi = np.mod(phi, 2 * np.pi)       # Convert φ to [0, 2π]

    # Bin the theta and phi values
    theta_bin_indices = np.digitize(theta, theta_edges) - 1
    phi_bin_indices = np.digitize(phi, phi_edges) - 1

    # Ensure bin indices are within valid range
    theta_bin_indices = np.clip(theta_bin_indices, 0, len(theta_edges) - 2)
    phi_bin_indices = np.clip(phi_bin_indices, 0, len(phi_edges) - 2)

    # Get per-point r_min and r_max from the r_stat maps
    if r_min_map is not None:
        r_min = r_min_map[theta_bin_indices, phi_bin_indices]
    else:
        r_min = None
    if r_max_map is not None:
        r_max = r_max_map[theta_bin_indices, phi_bin_indices]
    else:
        r_max = None

    # Clip radius
    if r_min is not None:
        r = np.maximum(r, r_min)
    if r_max is not None:
        r = np.minimum(r, r_max)

    # Convert back to Cartesian coordinates
    x_new = r * np.sin(theta) * np.cos(phi)
    y_new = r * np.sin(theta) * np.sin(phi)
    z_new = r * np.cos(theta)

    transformed_points = np.stack((x_new, y_new, z_new), axis=-1) + center
    return transformed_points.round()


# r_stat_center

def contour_transition(points_xyz, center_TC, r_min_map, r_max_map, lower_factor, upper_factor, num_points_to_modify = 1000, N=4096,anchor=None):

    theta_edges = np.linspace(0, np.pi, 129)  # 128 bins for theta
    phi_edges = np.linspace(0, 2 * np.pi, 129)  # 128 bins for phi

    N = points_xyz.shape[0]
    factor = np.random.uniform( lower_factor, upper_factor,size=num_points_to_modify)
    indices_to_modify = np.random.choice(N, num_points_to_modify, replace=False)

    points_modified = shrink_or_expand_points(points_xyz, center_TC, factor, indices=indices_to_modify)
    points_modified = clip_points_within_radius(points_modified, center_TC, theta_edges,phi_edges,r_min_map=r_min_map, r_max_map=r_max_map)
    
    if anchor.shape:
        d = (points_modified - anchor)/anchor
        distance = (((d)**2)**0.5).mean()
        
    return points_modified, distance

def contour_transition_equally(points_xyz, center_TC, r_min_map, r_max_map, factor, num_points_to_modify = 4096, N=4096, anchor=None):
    theta_edges = np.linspace(0, np.pi, 129)  # 128 bins for theta
    phi_edges = np.linspace(0, 2 * np.pi, 129)  # 128 bins for phi

    N = points_xyz.shape[0]
   
    indices_to_modify = np.random.choice(N, num_points_to_modify, replace=False)

    points_modified = shrink_or_expand_points(points_xyz, center_TC, factor, indices=indices_to_modify)
    points_modified = clip_points_within_radius(points_modified, center_TC,  theta_edges,phi_edges, r_min_map=r_min_map, r_max_map=r_max_map)
    
    if anchor.shape:
        d = (points_modified - anchor)/anchor.max()
        distance = (((d)**2)**0.5).mean()
        
    return points_modified, distance

def equally_sampling(given_points_xyz, center_TC, r_min_map, r_max_map, factor,num_ini = 10, num_samples_each = 10,withinTC=False, outWT=False):
    '''
    the equally sampling would expand/shirnk follow the countour shpae of TC/WT, all points move together
    
    
    
    #  TC shrink 
    sampling(TC_points_xyz,center_TC,r_stat_center,r_stat_TC, 0.95,withinTC=True)

    #  TC expand
    sampling(TC_points_xyz,center_TC,r_stat_TC,r_stat_WT, 1.05)

    #  WT shrink
    sampling(WT_points_xyz,center_TC,r_stat_TC,r_stat_WT, 0.95)

    #  WT expand
    sampling(WT_points_xyz,center_TC,r_stat_WT,r_stat_HT, 1.05,outWT=True)

    
    '''
    accept = []
    d_accept = []
    for i in range(num_ini):
        current_point = given_points_xyz
        for j in range(num_samples_each):

            current_point, distance = contour_transition_equally(current_point,center_TC,r_min_map, r_max_map, factor,anchor=given_points_xyz)
            accept.append(current_point)
            d_accept.append(distance)
    if withinTC:
        d_accept = np.ones(num_samples_each*num_ini)
    elif outWT:
         d_accept = np.zeros(num_samples_each*num_ini)
    return np.array(accept), np.array(d_accept)


def get_values_at_coords(image, coords):
    """
    Extracts the values from the image at the specified coordinates.

    Parameters:
    image (numpy.ndarray): The 3D image array with shape (240, 240, 155).
    coords (numpy.ndarray): An array of coordinates with shape (N, 3).

    Returns:
    numpy.ndarray: An array of values at the specified coordinates.
    """
    # Round and convert coordinates to integers
    coords = np.round(coords).astype(int)
    
    # Ensure coordinates are within the image bounds
    coords[:, 0] = np.clip(coords[:, 0], 0, image.shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image.shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, image.shape[2] - 1)
    
    # Extract values at the specified coordinates
    values = image[coords[:, 0], coords[:, 1], coords[:, 2]]
    return values

def get_patches_at_coords(image, coords, patch_size=(3, 3, 3)):
    """
    Extracts patches from the image at the specified coordinates.

    Parameters:
    image (numpy.ndarray): The 3D image array with shape (240, 240, 155).
    coords (numpy.ndarray): An array of coordinates with shape (N, 3).
    patch_size (tuple): The size of the patch to extract (default is (3, 3, 3)).

    Returns:
    numpy.ndarray: An array of patches with shape (N, patch_size[0], patch_size[1], patch_size[2]).
    """
    # Calculate half sizes
    half_sizes = [s // 2 for s in patch_size]
    
    # Round and convert coordinates to integers
    coords = np.round(coords).astype(int)
    
    patches = []
    for coord in coords:
        # Initialize start and end indices for each dimension
        starts = [coord[i] - half_sizes[i] for i in range(3)]
        ends = [coord[i] + half_sizes[i] + 1 for i in range(3)]
        
        # Handle edge cases by clipping to image boundaries
        slices = []
        pad_width = []
        for i in range(3):
            start = starts[i]
            end = ends[i]
            pad_before = pad_after = 0

            if start < 0:
                pad_before = -start
                start = 0
            if end > image.shape[i]:
                pad_after = end - image.shape[i]
                end = image.shape[i]
            
            slices.append(slice(start, end))
            pad_width.append((pad_before, pad_after))
        
        # Extract the patch
        patch = image[slices[0], slices[1], slices[2]]
        
        # Pad the patch if necessary
        if any(pad != (0, 0) for pad in pad_width):
            patch = np.pad(patch, pad_width, mode='constant', constant_values=0)
        
        patches.append(patch)
    
    return np.array(patches)

def equally_sampling_spherical(
    points_xyz,
    center,
    r_min_map,
    r_max_map,
    factor,
    num_iterations=50,
    theta_edges=None,
    phi_edges=None,
    withinTC=False,
    outWT=False
):
    """
    Perform iterative expansion or shrinkage in spherical coordinates, preserving zero padding.
    
    Parameters:
        points_xyz (np.ndarray): The original points in Cartesian coordinates, including zero padding.
        center (np.ndarray): The center point for the spherical coordinate transformation.
        r_min_map (np.ndarray): The minimum radius map.
        r_max_map (np.ndarray): The maximum radius map.
        factor (float): The expansion or shrinkage factor.
        num_iterations (int): Number of iterations to perform.
        theta_edges (np.ndarray): Edges defining the theta bins.
        phi_edges (np.ndarray): Edges defining the phi bins.
        withinTC (bool): Flag to set distance to ones.
        outWT (bool): Flag to set distance to zeros.
    
    Returns:
        accept (np.ndarray): The transformed points over all iterations.
                             Shape: (num_iterations, N, 3)
        d_accept (np.ndarray): The distance metrics over all iterations.
                               Shape: (num_iterations,)
    """

    if theta_edges is None or phi_edges is None:
        raise ValueError("theta_edges and phi_edges must be provided")

    N = points_xyz.shape[0]

    # Identify non-zero points (valid points)
    nonzero_mask = ~np.all(points_xyz == 0, axis=1)
    indices_nonzero = np.where(nonzero_mask)[0]
    num_nonzero = len(indices_nonzero)

    if num_nonzero == 0:
        raise ValueError("No non-zero points found in points_xyz.")

    # Shift non-zero points so center is at origin
    shifted_points = cartesian_to_spherical(points_xyz[nonzero_mask])

    # Convert to spherical coordinates once
    r_initial,theta, phi = shifted_points[:, 0], shifted_points[:, 1], shifted_points[:, 2]


    theta_bin_indices = np.digitize(theta, theta_edges) - 1
    phi_bin_indices = np.digitize(phi, phi_edges) - 1
    theta_bin_indices = np.clip(theta_bin_indices, 0, len(theta_edges) - 2)
    phi_bin_indices = np.clip(phi_bin_indices, 0, len(phi_edges) - 2)

    # Retrieve r_min and r_max using the corrected bin indices
    r_min = r_min_map[theta_bin_indices, phi_bin_indices]
    r_max = r_max_map[theta_bin_indices, phi_bin_indices]
    # Initialize r to r_initial
    r = r_initial.copy()

    # Prepare arrays to collect results
    accept = np.zeros((num_iterations, N, 3))
    d_accept = []

    # Perform iterative expansion/shrinkage
    for i in range(num_iterations):
        # Update radius
        # r = r * factor
   
        r = r * (1 + np.sign(factor)* np.abs(np.random.normal(0, np.abs(factor), r.shape)))
        # Clip radius
        r = np.clip(r, r_min, r_max)
        # Calculate distance for convergence (only for non-zero points)
        # distance = np.mean(np.abs(r - r_initial) / (r_initial + 1e-8))

        # d_ht = np.abs(r - r_max) 
        # d_tc = np.abs(r - r_min) 

        valid = r_min!=r_max
        if len(r[valid])>0:
            d_ht = np.abs(r[valid] - r_max[valid]) 
            d_tc = np.abs(r[valid] - r_min[valid]) 
            distance = d_ht / (d_ht + d_tc + 1e-8)
            distance = np.mean(distance)
        else:
            distance = 1
        # Store distance
        d_accept.append(distance)
        # Convert back to Cartesian coordinates
        x_new = r * np.sin(theta) * np.cos(phi)
        y_new = r * np.sin(theta) * np.sin(phi)
        z_new = r * np.cos(theta)
        transformed_points = np.stack((x_new, y_new, z_new), axis=-1) + center
        # Store transformed points in accept array at the correct indices
        accept[i, indices_nonzero, :] = transformed_points
        # Ensure that zero-padded points remain zeros
        accept[i, ~nonzero_mask, :] = 0.0

    # Convert d_accept to a NumPy array
    d_accept = np.array(d_accept)

    if withinTC:
        d_accept = np.ones_like(d_accept)
    elif outWT:
        d_accept = np.zeros_like(d_accept)

    return accept, d_accept

