import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import cv2

from models.unet import UNetResNet18_BN_Scaled, UNetResNet18_BN_Scaled28, ResNet34_BN_Scaled
from dataloader.dataloader import zDataLoader
from dataloader.dataset import SmokeCloud_Dataset_V2
from checkpoint.checkpoint import CheckpointMgr
from models.loss import FocalLoss_BCE
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='/project/data/smoke_cloud/result/')
    parser.add_argument('--output',default='./pth/smoke_cloud_unetr18_smoke100k_hw28/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=50,type=int)
    parser.add_argument('--batchsize',default=6,type=int)
    parser.add_argument('--view_interval',default=50,type=int)
    parser.add_argument('--ckpt_interval',default=1000,type=int)
    parser.add_argument('--loss',default='bce',type=str)
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--check', default=0, type=int)
    args = parser.parse_args()
    return args



"""
nohup python -u train_unetr18bn_smoke100k.py --batchsize 128  --lr 0.01 --mode 2 >x2smoke100k.out 2>&1 &
nohup python -u train_unetr18bn_smoke100k.py --batchsize 128 --loss focalloss --mode 2 >x2smoke100k_focal.out 2>&1 &
"""
def train(args):
    n_cuda_device = torch.cuda.device_count()
    if args.mode == 0:
        model = UNetResNet18_BN_Scaled(n_classes=1, use_bn=True)
        mask_size = (56, 56)
        args.output = './pth/smoke_cloud_unetr18_smoke100k/'
        print("UNetResNet18_BN_Scaled,", mask_size)
    elif args.mode == 1:
        model = UNetResNet18_BN_Scaled28(n_classes=1, use_bn=True)
        mask_size = (28,28)
        args.output = './pth/smoke_cloud_unetr18_smoke100k_hw28/'
        print("UNetResNet18_BN_Scaled28,", mask_size)
    elif args.mode == 2:
        model =ResNet34_BN_Scaled(n_classes=1, use_bn=True)
        mask_size = (14, 14)
        args.output = './pth/smoke_cloud_r34_smoke100k_hw14/'
        print("ResNet34_BN_Scaled,", mask_size)
    else:
        model = UNetResNet18_BN_Scaled(n_classes=1, use_bn=True)
        mask_size = (56,56)
        args.output = './pth/smoke_cloud_unetr18_smoke100k/'
        print("UNetResNet18_BN_Scaled,", mask_size)


    # n_cuda_device = 1
    dataset_train = SmokeCloud_Dataset_V2(s100kpath='/project/data/smoke_cloud/smoke100k/',
                                       ssspath='/project/data/smoke_cloud/smoke_images_with_annot/tagged/',
                                        mode='train', iter_times= int(1e5),
                                          mask_size=mask_size
                                          )
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize,workers_per_gpu=8 if n_cuda_device > 1 else 16,
                             num_gpus=n_cuda_device,dist=False,shuffle=True,pin_memory=True,verbose=True)(dataset_train)

    # model = UNetResNet18_BN_Scaled(n_classes=1, use_bn=True)
    # model = UNetResNet18_BN_Scaled28(n_classes=1, use_bn=True)
    model = model.initial()


    trainable_params = model.get_parameters()
    optimizer = torch.optim.SGD(trainable_params,lr=args.lr,momentum=0.9)
    # criterion = nn.BCEWithLogitsLoss() #(pos_weight=2.0*torch.ones(1))
    criterion = None
    if args.loss in ['default', 'bce']:
        criterion = nn.BCEWithLogitsLoss()  # (pos_weight=2.0*torch.ones(1))
    elif args.loss in ['focalloss']:
        criterion = FocalLoss_BCE(gamma=2.0, alpha=0.4)
    else:
        criterion = nn.BCEWithLogitsLoss()  # (pos_weight=2.0*torch.ones(1))
        print('criterion will use bce loss because {} is not known'.format(args.loss))

    print('criterion will use {}'.format(args.loss))

    checkpoint_op = CheckpointMgr(ckpt_dir=args.output)
    checkpoint_op.load_checkpoint(model,map_location='cpu')
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 30, 40], gamma=0.5,
                                                     last_epoch=-1)
    model = model.cuda()
    criterion = criterion.cuda()
    if n_cuda_device > 1:
        model = nn.DataParallel(model)
    model.train()


    for epoch in range(args.max_epoch):
        model.train()
        lr = optimizer.param_groups[0]['lr']
        epoch_loss, epoch_tm, iter_tm = [],0,[]
        tm = time.time()
        for ind,batch in enumerate(dataloader):
            process_line = ind/len(dataloader)*100
            imgs = batch['X']#[b,c,h,w]
            masks = batch['y']#[b,h,w]

            if ind == 0:
                print('X =', imgs.shape, 'y =', masks.shape)

            imgs = imgs.cuda()
            masks = masks.cuda()
            pred = model(imgs)#[b,h,w]

            loss = criterion(pred.reshape(-1),masks.reshape(-1))
            epoch_loss.append(loss.data.cpu().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tm = time.time() - tm
            epoch_tm += tm
            iter_tm.append(tm)
            if ind%args.view_interval==0:
                print('Epoch:{:3d}[{:.1f}%], iter:{}, loss:{:.3f}[{:.3f}], lr:{:.5f},{:.2f}s/iter'.format(
                    epoch, process_line ,ind, loss.item(), np.array(epoch_loss).mean(), lr ,np.array(iter_tm).mean()
                ))
            iter_tm = []
            tm = time.time()
        # outer

        print('Epoch:{:3d},total_loss: {:.3f}, lr:{:.5f},{:.2f}s'.format(
            epoch, np.array(epoch_loss).mean(), lr, epoch_tm
        ))
        scheduler.step()
        print('Saving...')
        checkpoint_op.save_checkpoint(model=model.module if n_cuda_device > 1 else model, verbose=False)
        # val(args)



def check(args):
    n_cuda_device = 1
    if args.mode == 0:
        model = UNetResNet18_BN_Scaled(n_classes=1, use_bn=True)
        mask_size = (56, 56)
        args.output = './pth/smoke_cloud_unetr18_smoke100k/'
        print("UNetResNet18_BN_Scaled,", mask_size)
    elif args.mode == 1:
        model = UNetResNet18_BN_Scaled28(n_classes=1, use_bn=True)
        mask_size = (28,28)
        args.output = './pth/smoke_cloud_unetr18_smoke100k_hw28/'
        print("UNetResNet18_BN_Scaled28,", mask_size)
    elif args.mode == 2:
        model =ResNet34_BN_Scaled(n_classes=1, use_bn=True)
        mask_size = (14, 14)
        args.output = './pth/smoke_cloud_r34_smoke100k_hw14/'
        print("ResNet34_BN_Scaled,", mask_size)
    else:
        model = UNetResNet18_BN_Scaled(n_classes=1, use_bn=True)
        mask_size = (56,56)
        args.output = './pth/smoke_cloud_unetr18_smoke100k/'
        print("UNetResNet18_BN_Scaled,", mask_size)

    model = model.initial()
    model = model.cuda()
    model.train()

    finput = torch.rand(8, 3, 224, 224).cuda()

    output = model(finput)

    print(output.shape)








if __name__ == '__main__':
    args = arg_parse()
    if args.check != 0:
        check(args)
    else:
        train(args)


