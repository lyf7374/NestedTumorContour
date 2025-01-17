
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import random
LABEL_NAME = ["bg", "NCR", "ED", "ET"]



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
        self.original_size = [240, 240]  # Assuming original image size
        self.crop_size = [224, 224]  # Crop size set to (224, 224)

        # Define the label mapping here
        self.class_mapping = {0: 0, 1: 1, 2: 2, 4: 3}  # Map class 4 to index 3

        self.dataset = []
        self.build_dataset()

    def build_dataset(self):
        for idx in range(len(self.t1_paths)):
            # Load the volumes
            t1_img = nib.load(self.t1_paths[idx]).get_fdata()
            t1gd_img = nib.load(self.t1gd_paths[idx]).get_fdata()
            t2_img = nib.load(self.t2_paths[idx]).get_fdata()
            flair_img = nib.load(self.flair_paths[idx]).get_fdata()
            label_img = nib.load(self.label_paths[idx]).get_fdata()

            # Assuming volumes are of shape [H, W, D]
            assert t1_img.shape == t1gd_img.shape == t2_img.shape == flair_img.shape == label_img.shape, \
                "All modalities must have the same shape"

            num_slices = t1_img.shape[2]  # D

            # For each slice, create a data point
            for slice_idx in range(num_slices):
                data_dict = {
                    't1': t1_img[:, :, slice_idx],
                    't1gd': t1gd_img[:, :, slice_idx],
                    't2': t2_img[:, :, slice_idx],
                    'flair': flair_img[:, :, slice_idx],
                    'label': label_img[:, :, slice_idx],
                    'slice_idx': slice_idx,
                    'patient_idx': idx
                }
                self.dataset.append(data_dict)

    def __len__(self):
        return len(self.dataset)

    def hwc_to_chw(self, img):
        img = np.float32(img)
        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        img = torch.from_numpy(img.copy())
        return img

    def center_crop(self, img, crop_size):
        h, w = img.shape[0], img.shape[1]
        ch, cw = crop_size
        start_h = (h - ch) // 2
        start_w = (w - cw) // 2
        return img[start_h:start_h+ch, start_w:start_w+cw]

    def random_crop(self, img, crop_size):
        h, w = img.shape[0], img.shape[1]
        ch, cw = crop_size
        if h == ch:
            start_h = 0
        else:
            start_h = random.randint(0, h - ch)
        if w == cw:
            start_w = 0
        else:
            start_w = random.randint(0, w - cw)
        return img[start_h:start_h+ch, start_w:start_w+cw]

    def perform_trans(self, img, mask):
        # Apply cropping
        if self.is_train:
            # Random crop during training
            img = self.random_crop(img, self.crop_size)
            mask = self.random_crop(mask, self.crop_size)
        else:
            # Center crop during validation/testing
            img = self.center_crop(img, self.crop_size)
            mask = self.center_crop(mask, self.crop_size)

        # Apply transforms if any
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img, mask

    def remap_labels(self, label):
        # Efficient remapping using numpy
        remapped_label = np.copy(label)
        for orig_label, new_label in self.class_mapping.items():
            remapped_label[label == orig_label] = new_label
        return remapped_label

    def __getitem__(self, index):
        data_dict = self.dataset[index]

        # Get the images and label
        t1 = data_dict['t1']
        t1gd = data_dict['t1gd']
        t2 = data_dict['t2']
        flair = data_dict['flair']
        label = data_dict['label']

        # Stack modalities to create a multi-channel image
        img = np.stack([t1, t1gd, t2, flair], axis=-1)  # [H, W, C]

        # Normalize images
        mean = np.mean(img, axis=(0, 1))
        std = np.std(img, axis=(0, 1))
        std[std == 0] = 1  # Avoid division by zero
        img = (img - mean) / std

        # Remap labels
        label = self.remap_labels(label)

        # Process transformations, including cropping
        img, label = self.perform_trans(img, label)

        # Convert to tensor
        img = self.hwc_to_chw(img)  # [C, H, W]
        label = torch.from_numpy(label).long()  # Assuming label is integer class labels

        data = {
            'img': img,
            'label': label,
            'slice_idx': data_dict['slice_idx'],
            'patient_idx': data_dict['patient_idx']
        }

        return data
