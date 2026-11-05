import torch.nn as nn
from thop import profile
import torch
import argparse
import os

from model.fsnet.FSNet_stem import FSNet

from model.comparison.kang_3dfv import Kang_3DFV
from model.comparison.resnet34 import ResNet34_FV
from model.comparison.mobilenetv2 import Mobilenetv2
from model.comparison.ghostnet import Ghostnet
from model.comparison.efficientnet import Efficientnet
from model.comparison.xception import Xception

# # ##################################################seting############################################################
parser = argparse.ArgumentParser()
parser.add_argument('-rs', '--results', type=str, default='PNF/', help='model save path', required=False)
parser.add_argument('-device', '--device', type=str, default='0', help='gpu index', required=False)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

# fsnet_comparison_ablation
[kang_3dfv, resnet, mobilenetv2, ghostnet, efficientnet, xception, fsnet_f, fsnet_s, fsnet_a, fsnet_fa, fsnet_sa, fsnet_x] = [Kang_3DFV(in_channels=1, num_classes=5000), ResNet34_FV(in_channels=1), Mobilenetv2(in_channels=1, num_classes=5000), Ghostnet(in_channels=1, num_classes=5000),Efficientnet(in_channels=1, num_classes=5000), Xception(in_channels=1, num_classes=5000),FSNet(in_channels=1, num_classes=5000, a_ablation=False, f_ablation=True, s_ablation=False, fa_ablation=False, sa_ablation=False, w_mode='x'), FSNet(in_channels=1, num_classes=5000, a_ablation=False, f_ablation=False, s_ablation=True, fa_ablation=False, sa_ablation=False, w_mode='x'), FSNet(in_channels=1, num_classes=5000, a_ablation=True, f_ablation=False, s_ablation=False, fa_ablation=False, sa_ablation=False, w_mode='x'), FSNet(in_channels=1, num_classes=5000, a_ablation=False, f_ablation=False, s_ablation=False, fa_ablation=True, sa_ablation=False, w_mode='x'), FSNet(in_channels=1, num_classes=5000, a_ablation=False, f_ablation=False, s_ablation=False, fa_ablation=False, sa_ablation=True, w_mode='x'), FSNet(in_channels=1, num_classes=5000, a_ablation=False, f_ablation=False, s_ablation=False, fa_ablation=False, sa_ablation=False, w_mode='x')]

def get_params_flops(list_model=None, list_name=None, path_rs=None, name_txt=None):
    if not os.path.exists(args.results):
        os.makedirs(args.results)
    list_rs = []
    x = torch.randn(1, 1, 224, 224)
    msg = ''
    for i, model in enumerate(list_model):
        print('model name:', list_name[i])
        
        if model == kang_3dfv:
            f = torch.randn(1, 256)
        if model == resnet:
            f = torch.randn(1, 512)
        elif model == mobilenetv2:
            f = torch.randn(1, 1280)
        elif model == ghostnet:
            f = torch.randn(1, 480)
        elif model == xception:
            f = torch.randn(1, 2048)
        elif model == efficientnet:
            f = torch.randn(1, 1280)
        elif model == fsnet_f:
            f = torch.randn(1, 256)
        elif model == fsnet_s:
            f = torch.randn(1, 256)
        elif model == fsnet_a:
            f = torch.randn(1, 256)
        elif model == fsnet_fa:
            f = torch.randn(1, 256)
        elif model == fsnet_sa:
            f = torch.randn(1, 256)
        elif model == fsnet_x:
            f = torch.randn(1, 256)
            
        else:
            print('no model name')
            
        flops, params = profile(model, inputs=(x,))
        flops_cls, params_cls = profile(model.f2c, inputs=(f,))
       
        msg += '\n' + list_name[i] + ' params: ' + str((params-params_cls)/1e6) + 'M, flops: ' + str((flops-flops_cls)/1e9) + 'G'
    with open(os.path.join(path_rs, name_txt), 'w') as f:
        f.write(msg)
        

get_params_flops(list_model=[kang_3dfv, resnet, mobilenetv2, ghostnet, efficientnet, xception, fsnet_f, fsnet_s, fsnet_a, fsnet_fa, fsnet_sa, fsnet_x] , list_name=['kang_3dfv', 'resnet', 'mobilenetv2', 'ghostnet', 'efficientnet', 'xception', 'fsnet_f', 'fsnet_s', 'fsnet_a', 'fsnet_fa', 'fsnet_sa', 'fsnet_x'], path_rs=args.results, name_txt='pnf_comparison_ablation.txt')


