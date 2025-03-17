import argparse
import os
import random
import numpy as np
import torch
from models.DCM_large import  CombinedModel,EncP,Img_CDK_encoder,ContourEncoder,train_contrastive_ranking_epoch_rank_order
from dataset.dataloader import HDF5BrainDataset_loadall
from torch.utils.data import random_split, DataLoader
from torch.optim import AdamW


parser = argparse.ArgumentParser(description="Hyperparameters for the Brain Tumor Segmentation Model")
parser.add_argument("--GPU_id", type=str, default="-1", help="ID for GPUs")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--para", type=bool, default=False, help="name add")

parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--model", type=int, default=0, help="model index")
parser.add_argument("--num_epoches", type=int, default=100, help="Number of epochs")
parser.add_argument("--n_layers", type=int, default=1, help="Number of epochs")
parser.add_argument("--mini_batch", type=int, default=1, help="Number of epochs")
parser.add_argument("--select_data", type=int, default=0, help="model index")
parser.add_argument("--pos", type=float, default=1.0, help="name add")
parser.add_argument("--neg", type=float, default=1.0, help="name add")
parser.add_argument("--inf", type=float, default=0.0, help="name add")
parser.add_argument("--sm", type=float, default=0.0, help="name add")
parser.add_argument("--dd", type=float, default=0.0, help="name add")
parser.add_argument("--alpha", type=float, default=1.0, help="Learning rate")
parser.add_argument("--tau", type=float, default=5.0, help="Learning rate")
args = parser.parse_args()

# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"

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

mini_batch = args.mini_batch
BATCH_SIZE =1 
beta1 = 0.5
beta2 = 0.999
para = args.para
tau = args.tau
pos=args.pos
neg=args.neg
inf=args.inf
sm=args.sm
dd=args.dd
select_dataset=args.select_data
n_layers =  args.n_layers
EPOCH = args.num_epoches
lr = args.lr
LR = lr
print('lr', lr, LR)




model_save_path = f'CM_large_{int(pos):1d}{int(neg):1d}_inf{int(inf*1):1d}_{int(dd*1):1d}.pth'

if n_layers !=1:
    model_save_path = model_save_path[:-4] + '_N{}'.format(n_layers) + model_save_path[-4:]
if tau!=5.0:
    model_save_path = model_save_path[:-4] + '_tau{}'.format(int(tau))+ model_save_path[-4:]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Initialize the dataset
if select_dataset ==0:
    # Initialize the dataset
    h5_data_dir = 'light_h5_data_WTonly'
    dataset = HDF5BrainDataset_loadall(h5_data_dir)

    # Define train-validation split sizes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use a fixed random seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
else:
    from torch.utils.data import Subset, ConcatDataset
    # Initialize the dataset
    h5_data_dir_1 = 'trainOnly_h5'
    h5_data_dir_2 = 'perfect_h5'

    # Create dataset instances from the directories.
    dataset_1 = HDF5BrainDataset_loadall(h5_data_dir_1)
    dataset_2 = HDF5BrainDataset_loadall(h5_data_dir_2)

    # Deterministic split for dataset_2:
    # - First 22 samples for training.
    # - Next 12 samples for validation.
    train_subset_dataset2 = Subset(dataset_2, list(range(20)))
    val_dataset = Subset(dataset_2, list(range(20, 31)))

    # Combine dataset_1 with the training subset of dataset_2.
    train_dataset = ConcatDataset([dataset_1, train_subset_dataset2])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)



pc_encoder = EncP(
    in_channels=3,            
    input_points=4096,
    num_stages=4,
    embed_dim=72,
    k_neighbors=30,
    alpha=100,
    beta=1000,
    LGA_block=[2,1,1,1],      
    dim_expansion=[2,2,2,2],  # changed last 1 -> 2
    type='mn40'
)
img_encoder = Img_CDK_encoder(embed_dim=32, output_dim=3, img_size=(128, 128, 128), patch_size=(4, 4, 4), in_chans=1, depths=[2, 2, 2], num_heads=[2, 4, 8, 16], window_size=(7, 7, 7), mlp_ratio=4.)
img_encoder.load_state_dict(torch.load("saved_models/CKD_best_model_BraTs.pkl", map_location='cpu'),strict=False)


contour_encoder = ContourEncoder(
    img_channels=1024,  # matching shape [B, 1024, ...]
    contour_dim=1152,   # matching your PC encoder's final dimension
    hidden_dim=256,
    n_heads=4,
    num_decoder_layers=1
)

model = CombinedModel(pc_encoder,img_encoder,contour_encoder)
if para and device_ids:
    print('ids', device_ids)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
if cuda:
    model.cuda()



# pcs = torch.rand(size = (10,3,4096)).cuda()
# img =torch.rand(size = (1,4,128,128,128)).cuda()
# for _ in tqdm.tqdm(range(100)):
#     # with torch.no_grad():
#     out =  model(pcs,img)

# print(out.shape)


optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    betas=(beta1, beta2),
    weight_decay=0.00005
)

def adjust_learning_rate(optimizer, LR, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = LR * ((1-epoch/EPOCH ).__pow__(0.9))
        param_group['lr'] = lr
        print('...working on epoch: {}/{} with learning rate: {:.7f}'.format(epoch,
              EPOCH , param_group['lr']))
        

# Initialize best validation loss
best_val_loss = float('inf')

early_stop = 0


for epoch in range(EPOCH):
    if (epoch+1) %10 ==0:
        adjust_learning_rate(optimizer, LR, epoch)  # adjust lr


    train_loss, test_loss = train_contrastive_ranking_epoch_rank_order(
        model=model,
        optimizer=optimizer,
        train_dataset=train_loader, 
        test_dataset=val_loader,
        epoch=epoch,     # pass the epoch index
        n_tc=100,
        n_inf=300,
        margin=0.0,
        tau=tau,
        lambda_pos=pos,
        lambda_neg=neg,
        lambda_inf=inf,
        lambda_dd=dd,
        mini_batch_size =  mini_batch
    )

    if test_loss is not None:
        if test_loss < best_val_loss:
            best_val_loss = test_loss
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_save_path)
            else:
                torch.save(model.state_dict(), model_save_path)
  
            loss_val = test_loss.item() if hasattr(test_loss, 'item') else test_loss
            print(f"# # # # # # # # # # # # Model saved at epoch {epoch + 1} with validation loss {loss_val:.6f}# # # # # # # # # # # # ")

            early_stop = 0  # Reset early stopping counter
        else:
            early_stop += 1
            print(f"# # # # # # # # # # # # No improvement in validation loss for {early_stop} epochs.# # # # # # # # # # # # ")
            if early_stop > 9:
                print("Early stopping triggered.")
                break
            # 
from utils.EvalSupport import  evaluate_dataset
_ = evaluate_dataset(val_loader,model)
print('current model', model_save_path)