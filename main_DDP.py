# CUDA_VISIBLE_DEVICES=0,1,2 python3 -m torch.distributed.launch --nproc_per_node=3 /home/External/xr/SDOAnet/doa_project/jyx/main_multiGPU.py
import torch
from torch import nn
from torchinfo import summary
import time
import  os
from network import ExMSELoss
from dataset import MatCovData, gen_loader
from collections import OrderedDict
from torchvision.models import resnet34
import  argparse
SAVE_PATH='/home/External/xr/SDOAnet/doa_project/jyx/model_res34_2ndTry'

0
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight) # an effective initialize strategy,control the variety of input and output
        # to ensure information can flow better when forward and back propagation


def vec_angle_err(y_hat, y):
    truth_vec = torch.stack((y[:, 1] * y[:, 3], y[:, 0] * y[:, 3], y[:, 2]), dim=1)
    pred_vec = torch.stack((y_hat[:, 1] * y_hat[:, 3], y_hat[:, 0] * y_hat[:, 3], y_hat[:, 2]), dim=1)
    inner_product = torch.sum(truth_vec * pred_vec, dim=1)
    module_length = torch.norm(truth_vec, dim=1) * torch.norm(pred_vec, dim=1)
    cos = torch.min(inner_product / module_length, torch.ones(y_hat.shape[0], dtype=y_hat.dtype, device=y_hat.device))
    # inner product could exceed module length.cannot understand why
    err = torch.mean(torch.rad2deg(torch.acos(cos))) # result in degree format is more readable
    return err


LAST_EPOCH=188

def train(net, epochs, train_loader,sampler, test_loader, optimizer, criterion, scheduler=None):
    tr_loss, te_loss, tr_err, te_err = [], [], [], []
    for e in range(LAST_EPOCH,epochs):

        epoch_start = time.time()
        sampler.set_epoch(e+int(epoch_start)%100)
        net.train()
        loss_train = 0
        angle_err_train = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            loss = criterion(y_hat, y[:, :4])
            loss.backward()
            optimizer.step()
            loss_train += loss.data.item()
            angle_err_train += vec_angle_err(y_hat.detach(), y[:, :4]).item()

        loss_train /= (i + 1) # it's surprising that i is not freed. Compilation mindset does not fit for interpretation.
        angle_err_train /= (i + 1)
        tr_loss.append(loss_train)
        tr_err.append(angle_err_train)

        net.eval()
        loss_test = 0
        angle_err_test = 0
        with torch.no_grad():
            for j, (x, y) in enumerate(test_loader):
                x, y = x.cuda(), y.cuda()
                y_hat = net(x)
                loss = criterion(y_hat, y[:, :4])
                loss_test += loss.data.item()
                angle_err_test += vec_angle_err(y_hat.detach(), y[:, :4]).item()

        loss_test /= (j + 1) # it's surprising that j is not freed
        angle_err_test /= (j + 1)
        te_loss.append(loss_test)
        te_err.append(angle_err_test)

        print("Epochs: %d / %d, Time: %.1f, "
              "Training Loss: %.5f, Train Err: %.5f, "
              "Validation Loss:  %.5f, Test Err: %.5f" %
              (e + 1, epochs, time.time() - epoch_start, loss_train, angle_err_train, loss_test, angle_err_test))
# formatting output,like printf in C language
        if scheduler:
            scheduler.step()

        if (e + 1) % 4 == 0 :
            torch.save(net.state_dict(), SAVE_PATH+'/epoch_{}.pth'.format(e + 1))
        # torch.save(net.state_dict(), SAVE_PATH + '/epoch_{}.pth'.format(e + 1))

    return net, (tr_loss, te_loss, tr_err, te_err)


if __name__ == '__main__':
    # 1) 初始化
    torch.distributed.init_process_group(backend="nccl")

    # 2） 配置每个进程的gpu
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'  # use all gpus
    # torch.cuda.empty_cache()
    # parser=argparse.ArgumentParser()
    # parser.add_argument("--local_rank",type=int)
    # args=parser.parse_args()

    epochs = 390
    lr = 0.0001
    batch_size = 512
    snr = [-5,10]

    in_channel = 2
    out_channels = [16, 64, 64, 128]
    in_len = 1000
    out_len = 4

    # with_tanh = True

    net = resnet34()
    # modify output of fully connection layer
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, out_len)
    net.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # rank should always be specified
    # torch.distributed.init_process_group(backend='nccl', init_method='file:///home/External/xr/SDOAnet/doa_project/jyx/some_file/file', world_size=3,rank=args.local_rank)

    net.to(device)
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank],  output_device=local_rank)
    net.load_state_dict(torch.load(SAVE_PATH + '/epoch_188.pth'))
    # net.load_state_dict(torch.load(SAVE_PATH + '/2nd_normed_epoch_'+str(LAST_EPOCH)+'.pth'))
    # net.apply(init_weights)  # recursively apply operation on the module and submodules

    # net = net.cuda()

    summary(net, input_size=(batch_size, in_channel, 5, in_len))
    # print info,necessary parameters:model,
    # input_size
    # optimizer = torch.optim.Adam([ {'params':filter(lambda p: p.requires_grad, net.parameters()),'initial_lr':0.01, 'lr':lr}])
    optimizer = torch.optim.Adam(         filter(lambda p: p.requires_grad, net.parameters()),  lr=lr)
    # only optimize the parameters that requires grad.lambda argument_list:expression filter(function,iterable)
    # optimizer=torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=lr,momentum=0.5,nesterov=True)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [12,30,80], gamma=0.2,last_epoch=LAST_EPOCH)
    # lr gets multiplied by gamma at milestones.
    criterion = ExMSELoss(reduction='mean')
    train_loader, test_loader,sampler = gen_loader(batch_size, snr,norm='norm')
    net, _ = train(net=net, epochs=epochs,
                   train_loader=train_loader, sampler=sampler,test_loader=test_loader,
                   optimizer=optimizer, criterion=criterion,
                    # scheduler=scheduler
                   )






# TODO 668500  796490
