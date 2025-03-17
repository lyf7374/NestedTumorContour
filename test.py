import argparse
import os
import random
import numpy as np
import torch
from models.DCM_large import  CombinedModel,EncP,Img_CDK_encoder,ContourEncoder
import tqdm 

parser = argparse.ArgumentParser(description="Hyperparameters for the Brain Tumor Segmentation Model")
parser.add_argument("--GPU_id", type=str, default="-1", help="ID for GPUs")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--para", type=bool, default=False, help="name add")


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


para = args.para
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Initialize the dataset

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
img_encoder.load_state_dict(torch.load("saved_model/CKD_best_model_BraTs.pkl", map_location='cpu'))


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


    pcs = torch.rand(size = (10,3,4096)).cuda()
    img =torch.rand(size = (1,4,128,128,128)).cuda()
for _ in tqdm.tqdm(range(100)):
    # with torch.no_grad():
    out =  model(pcs,img)

print(out.shape)