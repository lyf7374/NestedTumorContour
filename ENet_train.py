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
import torch.optim.lr_scheduler as lr_scheduler

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
    model_save_path = 'Enet_simple_neighbour.pth'
elif model_index ==1:
    model_save_path = 'Enet_neighbour.pth'
elif model_index ==2:
    model_save_path = 'Enet_sample.pth'
elif model_index ==3:
    model_save_path = 'Enet.pth'


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
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


from models.ENetModel import ENetSimple,ENet,train_neighbourwise_epoch,train_randompairs_epoch,initialize_weights

if model_index ==0 or model_index ==2:
    model = ENetSimple(
        contour_dim=1152,
        hidden_dim=256,
        num_layers=3,
        dropout=0.1
    )
elif model_index ==1 or model_index ==3:
    model = ENet(
        img_channels=1024,
        contour_dim=1152,
        hidden_dim=256,
        n_heads=4,
        num_decoder_layers=3
    )

initialize_weights(model)

# optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(beta1, beta2), weight_decay=0.00005)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    betas=(beta1, beta2),
    weight_decay=0.00005
)

# Learning rate scheduler: Reduce LR on plateau
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True
)


if con:
    dic_pth = model_save_path
    state_dict = torch.load(dic_pth, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
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
        train_loss, test_loss = train_neighbourwise_epoch(
            model=model,
            optimizer=optimizer,
            train_dataset=train_loader, 
            test_dataset=val_loader,
            epoch=epoch,        # pass the epoch index
            lr=lr
        )
    elif model_index == 2 or model_index == 3:
        train_loss, test_loss = train_randompairs_epoch(
        model=model,
        optimizer=optimizer,
        train_dataset=train_loader, 
        test_dataset=val_loader,
        epoch=epoch,        # pass the epoch index
        num_pairs=K,            # or any other subset size
        lr=lr
         )
    if test_loss is not None:
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1} with validation loss {test_loss:.6f}")
            early_stop = 0  # Reset early stopping counter
        else:
            early_stop += 1
            print(f"No improvement in validation loss for {early_stop} epochs.")
            if early_stop > 9:
                print("Early stopping triggered.")
                break