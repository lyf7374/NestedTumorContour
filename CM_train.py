import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import random_split, DataLoader
from utils.preprocessing_support import HDF5BrainDataset

# Step 1: Set up argparse
parser = argparse.ArgumentParser(description="Hyperparameters for the Brain Tumor Segmentation Model")

parser.add_argument("--GPU_id", type=str, default="-1", help="ID for GPUs")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--num_epoches", type=int, default=100, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--model", type=int, default=0, help="model index")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--para", type=bool, default=False, help="name add")
parser.add_argument("--con", type=bool, default=False, help="name add")
parser.add_argument("--eval", type=bool, default=False, help="name add")
parser.add_argument("--K", type=int, default=512, help="name add")
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

beta1 = 0.5
beta2 = 0.999


eval= args.eval
EPOCH = args.num_epoches
con = args.con
para = args.para
BATCH_SIZE = args.batch_size
model_index = args.model
lr = args.lr
K = args.K
LR = lr
print('lr', lr, LR)

if model_index == 0:
    model_save_path = 'pairwise_cat.pth'
elif model_index ==1:
    model_save_path = 'pairwise_cross.pth'
elif model_index ==2:
    model_save_path = 'listwise_cross.pth'
print('current model', model_save_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Initialize the dataset
h5_data_dir = 'light_h5_data'
dataset = HDF5BrainDataset(h5_data_dir)

# Define train-validation split sizes
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Use a fixed random seed for reproducibility
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

if model_index ==0:
    from models.ContrastiveModel import PairwiseContrastiveModel,train_pairwise_epoch
    model = PairwiseContrastiveModel(img_dim=1024, 
                                    contour_dim=1152,
                                    hidden_dim=256).to(device)   
elif model_index ==1:
    from models.ContrastiveModel import CrossAttnPairwiseModel,train_pairwise_epoch
    model = CrossAttnPairwiseModel(
                img_channels=1024,
                contour_dim=1152,
                hidden_dim=256,
                n_heads=4
            ).to(device)
elif model_index ==2:
    from models.ContrastiveModel import CrossAttnListwiseModel,train_lambdarank_epoch
    model = CrossAttnListwiseModel(hidden_dim=256, n_heads=4).to(device)


optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(beta1, beta2), weight_decay=0.00005)
if para and device_ids:
    print('ids', device_ids)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
if cuda:
    model.cuda()

# Initialize best validation loss
best_val_loss = float('inf')

early_stop = 0

for epoch in range(EPOCH):
    if model_index ==0 or model_index ==1:
        train_loss, test_loss = train_pairwise_epoch(
        model=model,
        optimizer=optimizer,
        train_dataset=train_loader, 
        test_dataset=val_loader,
        epoch=epoch,        # pass the epoch index
        num_pairs=K,            # or any other subset size
        lr=lr
         )
    elif model_index ==2:
        train_loss, test_loss = train_lambdarank_epoch(
            model=model,
            optimizer=optimizer,
            train_dataset=train_loader, 
            test_dataset=val_loader,
            epoch=epoch,        # pass the epoch index
            K=K,            # or any other subset size
            lr=lr
        )

    if test_loss:
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss {test_loss}")
        else:
            early_stop +=1
            if early_stop > 5:
                break