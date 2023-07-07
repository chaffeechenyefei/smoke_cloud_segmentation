import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import cv2

from models.unet import UNet2D
from dataloader.dataloader import zDataLoader
from dataloader.dataset import SmokeCloud_Dataset
from checkpoint.checkpoint import CheckpointMgr
from models.loss import FocalLoss_BCE
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',default='/project/data/coco2017/train2017')
    parser.add_argument('--pngpath',default='/project/data/smoke_cloud/smoke_png/')
    parser.add_argument('--result', default='/project/data/smoke_cloud/result/')
    parser.add_argument('--output',default='./pth/smoke_cloud_unet2d/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=100,type=int)
    parser.add_argument('--batchsize',default=6,type=int)
    parser.add_argument('--view_interval',default=50,type=int)
    parser.add_argument('--ckpt_interval',default=1000,type=int)
    args = parser.parse_args()
    return args


def val(args):
    print('#'*10, 'EVAL' , '#'*10)
    if not os.path.exists(args.result):
        os.makedirs(args.result, exist_ok=True)
    # n_cuda_device = torch.cuda.device_count()
    n_cuda_device = 1
    dataset_test = SmokeCloud_Dataset(bgpath=args.datapath, pngpath=args.pngpath,mode='test')
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize,workers_per_gpu=8 if n_cuda_device > 1 else 16,
                             num_gpus=n_cuda_device,dist=False,shuffle=False,pin_memory=True,verbose=True)(dataset_test)

    model = UNet2D(n_channels=3, n_classes=1)

    checkpoint_op = CheckpointMgr(ckpt_dir=args.output)
    checkpoint_op.load_checkpoint(model,map_location='cpu')

    model = model.cuda()
    if n_cuda_device > 1:
        model = nn.DataParallel(model)
    model.eval()

    cnt = 0
    for ind,batch in enumerate(dataloader):
        process_line = ind/len(dataloader)*100
        imgs = batch['X']#[b,c,h,w]
        ori_imgs = batch['img']#[b,h,w,c]

        imgs = imgs.cuda()
        preds = model.inference(imgs)#[b,h,w]

        for img,pred in zip(ori_imgs, preds):
            img = img.cpu().data.numpy().astype(np.uint8) #[h,w,3] uint8
            pred = pred.cpu().data.numpy() #[h,w] fp32

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[...,-1] = (hsv[...,-1]*pred).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            cv2.imwrite( pj(args.result,'{:04d}.png'.format(cnt)), img)

            cnt += 1
    print('#' * 10, 'EVAL END', '#' * 10)



"""
nohup python -u train_unet2d.py --batchsize 24 >x2smoke.out 2>&1 &
"""
def train(args):
    n_cuda_device = torch.cuda.device_count()
    # n_cuda_device = 1
    dataset_train = SmokeCloud_Dataset(bgpath=args.datapath, pngpath=args.pngpath,
                                       ssspath='/project/data/smoke_cloud/smoke_images_with_annot/tagged/',
                                        mode='train', iter_times= int(5e4))
    dataloader = zDataLoader(imgs_per_gpu=args.batchsize,workers_per_gpu=8 if n_cuda_device > 1 else 16,
                             num_gpus=n_cuda_device,dist=False,shuffle=True,pin_memory=True,verbose=True)(dataset_train)

    model = UNet2D(n_channels=3, n_classes=1)
    model = model.initial()


    trainable_params = model.get_parameters()
    optimizer = torch.optim.SGD(trainable_params,lr=args.lr,momentum=0.9)
    criterion = nn.BCEWithLogitsLoss() #(pos_weight=2.0*torch.ones(1))
    # criterion = FocalLoss_BCE(gamma=2.0, alpha=0.4)
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


