import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import cv2

from models.unet import UNet2D
from dataloader.dataloader import zDataLoader
from dataloader.dataset import SmokeCloud_Dataset_V2
from checkpoint.checkpoint import CheckpointMgr
from models.loss import FocalLoss_BCE
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='/project/data/smoke_cloud/result/')
    parser.add_argument('--output',default='./pth/smoke_cloud_unet2d_smoke100k/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=50,type=int)
    parser.add_argument('--batchsize',default=6,type=int)
    parser.add_argument('--view_interval',default=50,type=int)
    parser.add_argument('--ckpt_interval',default=1000,type=int)
    parser.add_argument('--loss',default='bce',type=str)
    args = parser.parse_args()
    return args



"""
nohup python -u train_unet2d_smoke100k.py --batchsize 24 >x2smoke100k.out 2>&1 &
nohup python -u train_unet2d_smoke100k.py --batchsize 24 --loss focalloss >x2smoke100k_focal.out 2>&1 &
"""
def train(args):
    n_cuda_device = torch.cuda.device_count()
    # n_cuda_device = 1
    dataset_train = SmokeCloud_Dataset_V2(s100kpath='/project/data/smoke_cloud/smoke100k/',
                                       ssspath='/project/data/smoke_cloud/smoke_images_with_annot/tagged/',
                                        mode='train', iter_times= int(1e5))
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize,workers_per_gpu=8 if n_cuda_device > 1 else 16,
                             num_gpus=n_cuda_device,dist=False,shuffle=True,pin_memory=True,verbose=True)(dataset_train)

    model = UNet2D(n_channels=3, n_classes=1)
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





if __name__ == '__main__':
    args = arg_parse()
    train(args)


