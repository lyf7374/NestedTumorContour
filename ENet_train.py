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
parser.add_argument("--margin", type=float, default=1.0, help="Learning rate")
parser.add_argument("--num_epoches", type=int, default=100, help="Number of epochs")
parser.add_argument("--n_layers", type=int, default=1, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument("--model", type=int, default=0, help="model index")
parser.add_argument("--h_dim", type=int, default=256, help="model index")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--para", type=bool, default=False, help="name add")
parser.add_argument("--con", type=bool, default=False, help="name add")
parser.add_argument("--eval", type=bool, default=False, help="name add")
parser.add_argument("--loop", type=bool, default=False, help="name add")
parser.add_argument("--skip", type=bool, default=False, help="name add")
parser.add_argument("--K", type=int, default=512, help="name add")
parser.add_argument("--A", type=float, default=0.0, help="name add")
parser.add_argument("--B", type=float, default=0.0, help="name add")
parser.add_argument("--C", type=float, default=0.0, help="name add")
parser.add_argument("--D", type=float, default=0.0, help="name add")
parser.add_argument("--alpha", type=float, default=0.5, help="Learning rate")
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
margin=args.margin
A=args.A
B=args.B
C=args.C
D=args.D
alpha = args.alpha

loop = args.loop
h_dim = args.h_dim
skip = args.skip
n_layers =  args.n_layers
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
    model_save_path = 'Enet_neighbour_L{}.pth'.format(n_layers)
elif model_index ==2:
    model_save_path = 'Enet_sample.pth'
elif model_index ==3:
    model_save_path = 'Enet_L{}.pth'.format(n_layers)
elif model_index ==4:
    model_save_path = 'Enet_multi{}.pth'.format(n_layers)
elif model_index ==5:
    model_save_path = 'Enet_her_multi.pth'
elif model_index ==6:
    model_save_path = 'Enet_con_simple.pth'
elif model_index ==7:
    model_save_path = 'Enet_con.pth'

    model_save_path = model_save_path[:-4] + '_K{}N{}'.format(K,n_layers) + model_save_path[-4:] 
elif model_index ==8:
    if (A==0.0) & (B==0.0) & (C==0.0) & (D==1.0):
        model_save_path = 'Enet_con_rank0001.pth'
    elif (A==0.0) & (B==1.0) & (C==1.0) & (D==1.0):
        model_save_path = 'Enet_con_rank0111.pth'
    else:
        model_save_path = 'Enet_con_rank.pth'   
elif model_index ==9:
    if (A==0.0) & (B==0.0) & (C==0.0) & (D==1.0):
        model_save_path = 'Enet_v2_con_rank0001.pth'
    elif (A==0.0) & (B==1.0) & (C==1.0) & (D==1.0):
        model_save_path = 'Enet_v2_con_rank0111.pth'
    else:
        model_save_path = 'Enet_v2_con_rank.pth'   

    if n_layers !=1 or K!=512:
        model_save_path = model_save_path[:-4] + '_K{}N{}'.format(K,n_layers) + model_save_path[-4:]

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



if model_index<4 or model_index ==6 or model_index ==7 or model_index ==8 or model_index ==9:
    from models.ENetModel import ENetSimple,ENet,ENet_v2, train_neighbourwise_epoch,train_randompairs_epoch,train_contrastive_epoch,train_contrastive_ranking_epoch
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
elif model_index ==4 or model_index ==5:
    from models.EnetMultiModel import CombinedBrainDataset,MultiScaleENet, HierarchicalMultiScaleENet, train_neighbourwise_epoch,train_randompairs_epoch

    # Initialize the dataset
    h5_img_dir = 'new_img_ht'
    img_dataset = HDF5BrainDataset(h5_img_dir)
    train_img, val_img = random_split(img_dataset, [train_size, val_size], generator=generator)

    combined_train_dataset = CombinedBrainDataset(train_dataset, train_img)
    combined_val_dataset = CombinedBrainDataset(val_dataset, val_img)

    train_loader  = DataLoader(combined_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(combined_val_dataset, batch_size=BATCH_SIZE, shuffle=False)

from models.ENetModel import initialize_weights

if model_index ==0 or model_index ==2 or model_index ==6:
    model = ENetSimple(
        contour_dim=1152,
        hidden_dim=h_dim,
        num_layers=3,
        dropout=0.1
    )
elif model_index ==1 or model_index ==3 or model_index ==7 or model_index ==8:
    model = ENet(
        img_channels=1024,
        contour_dim=1152,
        hidden_dim=h_dim,
        n_heads=4,
        num_decoder_layers=n_layers
    )
elif model_index ==9:
    model = ENet_v2(
        img_channels=1024,
        contour_dim=1152,
        hidden_dim=h_dim,
        n_heads=4,
        num_decoder_layers=n_layers
    )
elif model_index ==4:
    model= MultiScaleENet(

    )
elif model_index ==5:
    model=HierarchicalMultiScaleENet(

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
# scheduler = lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5, verbose=True
# )

def adjust_learning_rate(optimizer, LR, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = LR * ((1-epoch/EPOCH ).__pow__(0.9))
        param_group['lr'] = lr
        print('...working on epoch: {}/{} with learning rate: {:.7f}'.format(epoch,
              EPOCH , param_group['lr']))
        
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
    if (epoch+1) %10 ==0:
        adjust_learning_rate(optimizer, LR, epoch)  # adjust lr

    if model_index ==0 or model_index ==1:
        train_loss, test_loss = train_neighbourwise_epoch(
            model=model,
            optimizer=optimizer,
            train_dataset=train_loader, 
            test_dataset=val_loader,
            epoch=epoch,        # pass the epoch index
            lr=lr,
        
            skip_equal = skip
        )
    elif model_index == 2 or model_index == 3 or model_index ==4 or model_index ==5:
        train_loss, test_loss = train_randompairs_epoch(
        model=model,
        optimizer=optimizer,
        train_dataset=train_loader, 
        test_dataset=val_loader,
        epoch=epoch,        # pass the epoch index
        num_pairs=K,            # or any other subset size
        lr=lr,
        loop = loop,
        skip_equal = skip
         )
    elif model_index == 6 or model_index ==7:
        train_loss, test_loss = train_contrastive_epoch(
        model=model,
        optimizer=optimizer,
        train_dataset=train_loader, 
        test_dataset=val_loader,
        epoch=epoch,        # pass the epoch index
        num_pairs=K,            # or any other subset size
        lr=lr

         )
    elif model_index == 8 or model_index ==9:
        train_loss, test_loss = train_contrastive_ranking_epoch(
        model=model,
        optimizer=optimizer,
        train_dataset=train_loader, 
        test_dataset=val_loader,
        epoch=epoch,        # pass the epoch index
        num_pairs=K,            # or any other subset size
        lr=lr,
        alpha = alpha, 
        lambda_A=A,      # (1,1) or (0,0) – no ranking loss
        lambda_B=B,      # (1,0) or (0,1)
        lambda_C=C,      # (1,d) or (d,1)
        lambda_D=D       # (d1,d2)
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
print('current model', model_save_path)