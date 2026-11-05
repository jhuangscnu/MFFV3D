import os
import sys
sys.path.append(os.getcwd())
import argparse
SCRIPTS_DIR = os.getcwd()
sys.path.append(SCRIPTS_DIR)
import torch
import torch.optim as optim
import torchvision
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from loss.CenterLoss import CenterLoss
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.nn as nn
import numpy as np
import random
from labour.Fit import fit
from dataload.mffv_dataload import Get3DUnfoldTrainImageDirectly, Get3DUnfoldAuthenticationPairDirectly
from dataload.lfmb3dfb_dataload import GetTrainImage, GetAuthenticationPair
from model.fsnet.FSNet_stem import FSNet

# # ##################################################seting############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, default='fsnet_x', help='model and mode', required=True)
parser.add_argument('-rs', '--results', type=str, default='RS/test/', help='model save path', required=False)
parser.add_argument('-db', '--db_path', type=str, default='/path/to/the/dataset/', help='db path', required=False) # sha
parser.add_argument('-device', '--device', type=str, default='0', help='gpu index', required=False)

parser.add_argument('-ep', '--epochs', type=int, default=2000, help='training epoch', required=False)
# parser.add_argument('-d', '--dataset', type=str, default='MFFV-N-3DUnfold', help='name of dataset', required=False)
parser.add_argument('-pt', '--protocol_path', type=str, default='pt/', help='protocol file path', required=False)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
# python train_lfmb3dfb.py -m fsnet_x -rs RS/lfmb3dfb/trained_model/tex/fvfs_x/ -db /home/duan/dataset/LFMB-3DFB_ACEUnfold/tex/
####################################################for fix random seed################################################
seed = 4
os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
def seed_torch(seed):
	  random.seed(seed)
	  os.environ['PYTHONHASHSEED'] = str(seed) 
	  np.random.seed(seed)
	  torch.manual_seed(seed)
	  torch.cuda.manual_seed(seed)
	  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True
seed_torch(seed)
def worker_init_fn(worker_id):
    np.random.seed(seed)
    random.seed(seed+worker_id)
g = torch.Generator()
g.manual_seed(seed)

## for data augmentation
train_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
])

test_data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
])


# for dataloader
dataset_train = GetTrainImage(path_db=args.db_path, path_csv=args.protocol_path, csv_name='lfmb3dfb_train.csv', postfix='.bmp', transform=train_data_transform)
dataset_dev = GetAuthenticationPair(path_db=args.db_path, path_csv=args.protocol_path, csv_name='lfmb3dfb_dev.csv', postfix='.bmp', transform=test_data_transform)
loader_train = torch.utils.data.DataLoader(dataset = dataset_train, batch_size=8, shuffle=True, num_workers=8, worker_init_fn=worker_init_fn, generator=g)
loader_dev = torch.utils.data.DataLoader(dataset = dataset_dev, batch_size=8, shuffle=False, num_workers = 8, worker_init_fn=worker_init_fn, generator=g)


# for losses
ce_loss = nn.CrossEntropyLoss()


# for model
if args.model == 'fsnet_x':
    model = FSNet(in_channels=1, num_classes=5000, a_ablation=False, f_ablation=False, s_ablation=False, fa_ablation=False, sa_ablation=False, w_mode='x')
    ct_loss = center_loss = CenterLoss(num_classes=5000, feat_dim=256)
else:
    print('please provide the correct model name! \n' * 10)
# for optimizer
lr = 0.001
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)#
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=0.00000001, last_epoch=-1) #

def check_files(folder_path, extension):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for file in os.listdir(folder_path):
        if file.endswith(extension):
            return True
    return False

def get_max_file(folder_path):
    files = os.listdir(folder_path)
    files = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    return files[-1]

if __name__ == '__main__':
    if torch.cuda.is_available():
        print('running with gpu')
        ce_loss.cuda()
        ct_loss.cuda()
        model.cuda()
    else:
        print('running with cpu')
        
    if check_files(args.results+'continue_model/', '.ckpt'):
        print('there are some ckpt')
        last_ckpt = get_max_file(args.results+'continue_model/')
        model.load_state_dict(torch.load(args.results+'continue_model/'+last_ckpt))
        last_epoch = int(last_ckpt[:-5])
        print('stating with epoch:', last_epoch)
    else:
        print('there is no ckpt')
        last_epoch = 0

    fit(args=args,
    train_loader=loader_train,
    dev_loader=loader_dev,
    model=model,
    loss_fn1=ce_loss,
    beta=0.01,
    loss_fn2=ct_loss,
    optimizer=optimizer,
    scheduler=scheduler,
    n_epochs=args.epochs-last_epoch,
    last_epoch = last_epoch,
    model_save_path=args.results,
    print_label=args.model,)



