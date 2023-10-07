import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from scipy import io
# import h5py
import mat73
# import hdf5storage

PATH='/home/External/xr/SDOAnet/doa_project/experiment'
# PATH='C:/Users/1/Desktop/fsdownload'
class MatCovData(Dataset):
    def __init__(self, snr=[-5,10], mode='train'):
        super(MatCovData, self).__init__()
        if mode == 'train':
            file_path=[]
            for i in range(0,len(snr)):
                file_path.append(PATH+'/train/train_e_simu_data_snr_{}.mat'.format(snr[i]))
        elif mode == 'validate':
            file_path = PATH+'/test/test_e_simu_data_snr_{}.mat'.format(snr)
        elif mode == 'test':
            file_path = PATH+'/a/data_snr_{}.mat'.format(snr)
        # mat_data=h5py.File(file_path,'r')
        # mat_data = io.loadmat(file_path)
        data_mats=[]
        for i in range(0, len(snr)):
            data_mats.append( mat73.loadmat(file_path[i]))
        # mat_data = hdf5storage.loadmat(file_path)
        alpha,beta,x_in_ls=[],[],[]
        for i in range(0, len(snr)):
            alpha.append(data_mats[i]['alpha_ls'])
            beta.append(data_mats[i]['beta_ls'])
            x_in_ls.append( np.transpose(data_mats[i]['X_in_ls'], (2, 0, 1)))
        alpha = np.concatenate(alpha, axis=0)
        beta = np.concatenate(beta, axis=0)
        x_in_ls = np.concatenate(x_in_ls, axis=0)
        self.total_num = alpha.size

        sin_alpha = np.sin(np.deg2rad(alpha))
        cos_alpha = np.cos(np.deg2rad(alpha))
        sin_beta = np.sin(np.deg2rad(beta))
        cos_beta = np.cos(np.deg2rad(beta))

        self.input = torch.from_numpy(np.stack((np.real(x_in_ls), np.imag(x_in_ls)), axis=1)).float()

        self.output = torch.from_numpy(np.stack((sin_alpha, cos_alpha, sin_beta, cos_beta, alpha, beta), axis=-1)).float()

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        return self.input[item], self.output[item]


def gen_loader(batch_size=512, snr=30):
    train_loader = DataLoader(MatCovData(snr=snr, mode='train'), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MatCovData(snr=snr, mode='validate'), batch_size=batch_size, shuffle=True)
    return train_loader, test_loader