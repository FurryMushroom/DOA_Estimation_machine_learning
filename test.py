import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import time
import  os
input_size = 1
output_size = 2
batch_size = 6
data_size = 24

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
    # 我们的模型

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print(
              "input", input)

        return output

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # use all gpus
    torch.cuda.empty_cache()
    net =Model(input_size,output_size)
    net=torch.nn.DataParallel(net)
    net=net.cuda()

    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                             batch_size=batch_size, shuffle=True)
    epoches=10
    for e in range(epoches):
     for data in rand_loader:
         print('epochs:',e)
         input = data.cuda()
         output = net(input)
