import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, ratio=8):
        super(ChannelAttention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1) # parameter:output size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # using conv2d replace shared mlp which used in channel dimension
        self.shared_mlp = nn.Sequential(nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1, bias=True),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channel // ratio, in_channel, kernel_size=1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.shared_mlp(self.max_pool(x))# squeeze operation:use both avg pooling and max pooling
        #use mlp instead of fully connection to do excitation
        #maybe this is because the question aims at every channel,or every point?
        avg_out = self.shared_mlp(self.avg_pool(x))
        return self.sigmoid(max_out + avg_out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat((avg_out, max_out), dim=1))) * x


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel,
                              padding='same', padding_mode='circular',
                              bias=bias)
        self.bn = nn.BatchNorm2d(out_channel)
        self.active = nn.ReLU()

    def forward(self, x):
        return self.active(self.bn(self.conv(x)))


class MainNet(nn.Module):
    def __init__(self, in_channel, out_channels, in_len, out_len=4, with_tanh=False):
        super(MainNet, self).__init__()
        convs = []
        in_ = in_channel
        for i, out_ in enumerate(out_channels):
            convs.append(ConvBlock(in_, out_, kernel=5))
            convs.append(nn.MaxPool2d((1, 3)))
            in_ = out_
            in_len //= 3
        convs.append(SpatialAttention())
        convs.append(ChannelAttention(in_))
        # convs.append(nn.AdaptiveAvgPool2d((5, 1)))
        self.conv_blocks = nn.Sequential(*convs)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(in_ * in_len * 5, 1024, bias=True),
                                nn.ReLU(),
                                nn.Linear(1024, in_, bias=True),
                                nn.ReLU(),
                                nn.Linear(in_, out_len, bias=True))
        self.tanh = nn.Tanh() if with_tanh else None

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.flatten(x)
        y = self.fc(x)
        if self.tanh is not None:
            y = self.tanh(y)
        return y


class ExMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ExMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_hat, y):
        if len(y_hat.shape) == 1:
            y_hat.reshape(1, -1)

        sqr_y_hat = torch.square(y_hat)
        sum_mse = torch.sum(torch.square(y_hat - y), dim=-1)
        punish = torch.square(1 - sqr_y_hat[:, 0] - sqr_y_hat[:, 1]) \
                 + torch.square(1 - sqr_y_hat[:, 2] - sqr_y_hat[:, 3])
        loss = sum_mse + punish
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.sqrt(torch.mean(loss))
        elif self.reduction == 'sum':
            return torch.sqrt(torch.sum(loss))


class CFEBlock(nn.Module):
    def __init__(self, in_channel):
        super(CFEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding='same')
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding='same')
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=1, padding='same')
        self.bn3 = nn.BatchNorm2d(32)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.max_pool(x)
        y = self.relu(self.bn3(self.conv3(x)))
        return y


class MSFPBlock(nn.Module):
    def __init__(self):
        super(MSFPBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.relu = nn.ReLU()

        self.parallel_conv1 = nn.Conv2d(32, 32, kernel_size=1, padding='same')
        self.parallel_bn1 = nn.BatchNorm2d(32)
        self.parallel_conv2 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding='same')
        self.parallel_bn2 = nn.BatchNorm2d(32)
        self.parallel_conv3 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding='same')
        self.parallel_bn3 = nn.BatchNorm2d(32)
        self.parallel_conv4 = nn.Conv2d(32, 32, kernel_size=(5, 1), padding='same')
        self.parallel_bn4 = nn.BatchNorm2d(32)

        self.res_conv = nn.Conv2d(32, 32, kernel_size=1, padding='same')
        self.res_bn = nn.BatchNorm2d(32)

        self.concat_conv = nn.Conv2d(32 * 4, 32, kernel_size=1, padding='same')
        self.concat_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.max_pool(x)
        res_x = self.res_bn(self.res_conv(x))
        parallel_x1 = self.relu(self.parallel_bn1(self.parallel_conv1(x)))
        parallel_x2 = self.relu(self.parallel_bn2(self.parallel_conv2(x)))
        parallel_x3 = self.relu(self.parallel_bn3(self.parallel_conv3(x)))
        parallel_x4 = self.relu(self.parallel_bn4(self.parallel_conv4(x)))
        concat_x = self.concat_bn(self.concat_conv(torch.cat((parallel_x1, parallel_x2, parallel_x3, parallel_x4), dim=1)))
        y = torch.add(res_x, concat_x)
        return y


class CATBlock(nn.Module):
    def __init__(self):
        super(CATBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))
        self.relu = nn.ReLU()

        self.group_conv1 = nn.Conv2d(32, 32, kernel_size=(1, 3), padding='same', groups=8)
        self.group_bn1 = nn.BatchNorm2d(32)
        self.group_conv2 = nn.Conv2d(32, 32, kernel_size=(3, 1), padding='same', groups=8)
        self.group_bn2 = nn.BatchNorm2d(32)

        self.res_conv = nn.Conv2d(32, 32, kernel_size=1, padding='same')
        self.res_bn = nn.BatchNorm2d(32)

        self.concat_conv = nn.Conv2d(32 * 2, 32, kernel_size=1, padding='same')
        self.concat_bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.max_pool(x)
        res_x = self.res_bn(self.res_conv(x))
        group_x1 = self.relu(self.group_bn1(self.group_conv1(x)))
        group_x2 = self.relu(self.group_bn2(self.group_conv2(x)))
        concat_x = self.concat_bn(self.concat_conv(torch.cat((group_x1, group_x2), dim=1)))
        y = torch.add(res_x, concat_x)
        return y


class AnotherNet(nn.Module):
    def __init__(self, in_channel, out_len=4, with_tanh=False):
        super(AnotherNet, self).__init__()
        self.cfe = CFEBlock(in_channel)
        self.msfp1 = MSFPBlock()
        self.msfp2 = MSFPBlock()
        self.cat1 = CATBlock()
        self.cat2 = CATBlock()
        self.cat3 = CATBlock()

        self.conv = nn.Conv2d(32, 32, kernel_size=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.global_avg_pool = nn.AdaptiveAvgPool2d((5, 1))

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(32 * 5, 32, bias=True),
                                nn.ReLU(),
                                nn.Linear(32, out_len, bias=True))
        self.tanh = nn.Tanh() if with_tanh else None

    def forward(self, x):
        x = self.cfe(x)
        x = self.msfp2(self.msfp1(x))
        x = self.cat3(self.cat2(self.cat1(x)))
        x = self.relu(self.bn(self.conv(x)))
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        y = self.fc(x)
        
        if self.tanh:
            y = self.tanh(y)

        return y
