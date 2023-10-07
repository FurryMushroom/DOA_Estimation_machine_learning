import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import mat73


class MatCovData(Dataset):
    def __init__(self, snr=10, mode='train', norm=None):
        super(MatCovData, self).__init__()
        if mode == 'train':
            file_path = '/home/External/xr/SDOAnet/doa_project/experiment/train/train_e_simu_data_snr_{}.mat'.format(snr)
        elif mode == 'validate':
            file_path = '/home/External/xr/SDOAnet/doa_project/experiment/test/test_e_simu_data_snr_{}.mat'.format(snr)
        elif mode == 'test':
            file_path = '/home/External/xr/SDOAnet/doa_project/experiment/a/data_snr_{}.mat'.format(snr)
        mat_data = mat73.loadmat(file_path)
        alpha = mat_data['alpha_ls']
        beta = mat_data['beta_ls']
        x_in_ls = np.transpose(mat_data['X_in_ls'], (2, 0, 1))

        self.total_num = alpha.size

        sin_alpha = np.sin(np.deg2rad(alpha))
        cos_alpha = np.cos(np.deg2rad(alpha))
        sin_beta = np.sin(np.deg2rad(beta))
        cos_beta = np.cos(np.deg2rad(beta))
        if norm == 'norm':
            x_norm = x_in_ls / np.sqrt(np.square(np.linalg.norm(x_in_ls[:, 0,:], 2, -1).reshape(-1, 1, 1)) / 1000)
            self.input = torch.from_numpy(np.stack((np.real(x_norm), np.imag(x_norm)), axis=1)).float()
        elif norm == 'Max':
            x_Max = x_in_ls / np.max(np.abs(x_in_ls), axis=(1, 2)).reshape(-1, 1, 1)
            self.input = torch.from_numpy(np.stack((np.real(x_Max), np.imag(x_Max)), axis=1)).float()
        else:
            self.input = torch.from_numpy(np.stack((np.real(x_in_ls), np.imag(x_in_ls)), axis=1)).float()

        self.output = torch.from_numpy(np.stack((sin_alpha, cos_alpha, sin_beta, cos_beta, alpha, beta), axis=-1)).float()

    def __len__(self):
        return self.total_num

    def __getitem__(self, item):
        return self.input[item], self.output[item]


def gen_loader(batch_size=512, snr=30, norm=None):
    if isinstance(snr, int):
        train_set = MatCovData(snr=snr, mode='train', norm=norm)
        test_set = MatCovData(snr=snr, mode='validate', norm=norm)
    elif isinstance(snr, list):
        train_sets = []
        test_sets = []
        for s in snr:
            train_sets.append(MatCovData(snr=s, mode='train', norm=norm))
            test_sets.append(MatCovData(snr=s, mode='validate', norm=norm))
        train_set = ConcatDataset(train_sets)
        test_set = ConcatDataset(test_sets)
    else:
        train_set = MatCovData(mode='train', norm=norm)
        test_set = MatCovData(mode='validate', norm=norm)
    sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    train_loader = DataLoader(train_set,
                              batch_sampler=sampler,
                              batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader\
        ,sampler

