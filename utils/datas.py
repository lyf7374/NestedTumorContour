import nibabel as nib
import os
import numpy as np

import numpy as np
import random
import torch
from scipy import ndimage
from torchvision.transforms import transforms

final_shape = 128


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')
        return None

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
            
def process_nii(patient_path, has_label=True):
    
    for file in os.listdir(patient_path):

        file_path = os.path.join(patient_path, file)
        if file.endswith('_flair.nii.gz'):
            flair_paths = file_path
        elif file.endswith('_t1.nii.gz'):
            t1_paths = file_path
        elif file.endswith('_t1ce.nii.gz'):
            t1ce_paths = file_path
        elif file.endswith('_t2.nii.gz'):
            t2_paths = file_path
        elif file.endswith('_seg.nii.gz'):
            label_paths  = file_path
        
    if has_label:
        label = np.array(nib_load(label_paths), dtype='uint8', order='C')
        images =np.stack([np.array(nib_load(t1_paths), dtype='float32'),
                 np.array(nib_load(t1ce_paths), dtype='float32'),
                 np.array(nib_load(t2_paths), dtype='float32'),
                 np.array(nib_load(flair_paths), dtype='float32')], -1)   # [240,240,155]
        mask = images.sum(-1) > 0
     
    for k in range(4):
        x = images[..., k]  #
        y = x[mask]
       
        # 0.8885
        x[mask] -= y.mean()
        x[mask] /= y.std()

        images[..., k] = x
    return images,label




class MaxMinNormalization(object):
    def __call__(self, img, label):
        Max = np.max(img)
        Min = np.min(img)
        img = (img - Min) / (Max - Min)
        return img, label

class Random_Flip(object):
    def __call__(self, img, label):
        if random.random() < 0.5:
            img = np.flip(img, 0)
            label = np.flip(label, 0)
        if random.random() < 0.5:
            img = np.flip(img, 1)
            label = np.flip(label, 1)
        if random.random() < 0.5:
            img = np.flip(img, 2)
            label = np.flip(label, 2)
        return img, label

class Random_Crop(object):
    def __call__(self, img, label):
        H = random.randint(0, 240 - final_shape)
        W = random.randint(0, 240 - final_shape)
        D = random.randint(0, 155 - final_shape)
        img = img[H: H + final_shape, W: W + final_shape,  D: D + final_shape , ...]
        label = label[..., H: H + final_shape, W: W + final_shape,  D: D + final_shape  ]
        return img, label
class Center_Crop(object):
    def __call__(self, img, label):
        # Calculate center crop dimensions
        H_center = (240 - final_shape) // 2
        W_center = (240 - final_shape) // 2
        D_center = (155 - final_shape) // 2

        # Perform the center crop
        img = img[H_center: H_center + final_shape, W_center: W_center + final_shape,  D_center: D_center + final_shape, ...]
        label = label[..., H_center: H_center + final_shape, W_center: W_center + final_shape,  D_center: D_center + final_shape]

        return img, label

class Random_intencity_shift(object):
    def __call__(self, img, label, factor=0.1):
        scale_factor = np.random.uniform(1.0-factor, 1.0+factor, size=[1, img.shape[1], 1, img.shape[-1]])
        shift_factor = np.random.uniform(-factor, factor, size=[1, img.shape[1], 1, img.shape[-1]])
        img = img*scale_factor+shift_factor
        return img, label

class Random_rotate(object):
    def __call__(self, img, label):
        angle = round(np.random.uniform(-10, 10), 2)
        img = ndimage.rotate(img, angle, axes=(0, 1), reshape=False)
        label = ndimage.rotate(label, angle, axes=(0, 1), reshape=False)
        return img, label

class Pad(object):
    def __call__(self, img, label):
        img = np.pad(img, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        label = np.pad(label, ((0, 0), (0, 0), (0, 5)), mode='constant')
        return img, label

class ToTensor(object):
    def __call__(self, img, label):
        img = np.ascontiguousarray(img.transpose(3, 0, 1, 2))
        label = np.ascontiguousarray(label)
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).long()
        return img, label

def transform(img, label):

    pad = Pad()
    random_crop = Random_Crop()
    random_flip = Random_Flip()
    center_crop = Center_Crop()
    random_intensity_shift = Random_intencity_shift()
    to_tensor = ToTensor()

    img, label = pad(img, label)
    img, label = random_crop(img, label)

#     img, label = center_crop(img, label)
    img, label = random_flip(img, label)
    img, label = random_intensity_shift(img, label)
    img, label = to_tensor(img, label)

    return img, label



