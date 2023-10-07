import numpy as np
import torch
from torch import nn
import mat73
import scipy.io as scio
from network import MainNet, ExMSELoss
from dataset import MatCovData
from collections import OrderedDict
from torchvision.models import resnet50

SAVE_PATH = '/home/External/xr/SDOAnet/doa_project/jyx/model_res50'


# SAVE_PATH='C:/Users/1/Desktop/fsdownload/resnet_model144.pth'

def pred(net, data, device):
    net.eval()
    with torch.no_grad():
        x, y = data[0].to(device), data[1].to(device)
        truth = torch.stack((y[:, 1] * y[:, 3], y[:, 0] * y[:, 3], y[:, 2]), dim=1)
        y_hat = net(x)
        pred = torch.stack((y_hat[:, 1] * y_hat[:, 3], y_hat[:, 0] * y_hat[:, 3], y_hat[:, 2]), dim=1)
    return pred, truth


def calc_vec_err(vec_hat, vec, reduction='mean'):
    inner_product = torch.sum(vec * vec_hat, dim=1)
    module_length = torch.norm(vec, dim=1) * torch.norm(vec_hat, dim=1)
    cos = torch.min(inner_product / module_length, torch.ones(vec.shape[0], dtype=vec.dtype, device=vec.device))
    if reduction == 'mean':
        err = torch.mean(torch.rad2deg(torch.acos(cos)))
    elif reduction == 'none':
        err = torch.rad2deg(torch.acos(cos))
    else:
        err = None
    return err


if __name__ == '__main__':
    device = torch.device('cpu')
    # SNR = [0, 5, 10, 15, 20, 25, 30]
    SNR = [-10,-5,0,5,10,15,20]
    in_channel = 2
    out_channels = [16, 64, 64, 128]
    in_len = 1000
    out_len = 4

    with_tanh = False

    net = resnet50()
    # modify output of fully connection layer
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, out_len)
    net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    err_ls = []
    for snr in SNR:
        print('testing in {} signal...'.format(snr))
        net.load_state_dict(torch.load(SAVE_PATH + '/normed_epoch_64.pth'))
        net.to(device)
        dataset = MatCovData(snr=snr, mode='test',norm='norm')
        pred_vec, truth_vec = pred(net, dataset[:], device)
        err=calc_vec_err(pred_vec, truth_vec).item()
        print(err)
        err_ls.append(err)
    # scio.savemat(SAVE_PATH+'/pred_result/normed_res50.mat', {'obj_ls': np.array(err_ls)})
# TODO 3271355
