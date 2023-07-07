import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import cv2

from models.unet import UNet2D
from dataloader.dataloader import zDataLoader
from dataloader.dataset import Basic_Dataset
from checkpoint.checkpoint import CheckpointMgr
from models.loss import FocalLoss_BCE
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',default='/project/data/Fire/smoke_ustc/RF_dataset/JPEGImages/')
    parser.add_argument('--result', default='/project/data/smoke_cloud/result/')
    parser.add_argument('--output',default='./pth/smoke_cloud_unet2d_smoke100k/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=100,type=int)
    parser.add_argument('--batchsize',default=8,type=int)
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
    dataset_test = Basic_Dataset(datapath=args.datapath)
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
        print('{:.2f}% done'.format(process_line), end='\r')
        if process_line > 10:
            break
        imgs = batch['X']#[b,c,h,w]
        ori_imgs = batch['img']#[b,h,w,c]

        imgs = imgs.cuda()
        preds = model.inference(imgs)#[b,h,w]

        for img,pred in zip(ori_imgs, preds):
            img = img.cpu().data.numpy().astype(np.uint8) #[h,w,3] uint8
            pred = pred.cpu().data.numpy().squeeze() #[h,w] fp32

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[...,-1] = (hsv[...,-1]*pred).astype(np.uint8)
            # h_ch = 1*hsv[...,0]
            # # print(hsv.shape, h_ch.shape, pred.shape)
            # h_ch[pred>0.5] = 0
            # hsv[...,0] = h_ch.astype(np.uint8)
            img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            img = np.concatenate([img,img2], axis=1)

            # r_ch = img[...,-1]*1
            # # upred = pred*255
            # # r_ch[pred>0.1] = upred[pred>0.1]
            # r_ch[pred>0.3] = 255
            # img[...,-1] = r_ch.astype(np.uint8)

            cv2.imwrite( pj(args.result,'{:04d}.png'.format(cnt)), img)

            cnt += 1
    print('#' * 10, 'EVAL END', '#' * 10)






if __name__ == '__main__':
    args = arg_parse()
    val(args)


