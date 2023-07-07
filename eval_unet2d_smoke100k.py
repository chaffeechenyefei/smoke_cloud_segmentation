import os
import time
import numpy as np
import argparse
import torch
import torch.nn as nn
import cv2

from models.unet import UNet2D
from dataloader.dataloader import zDataLoader
from dataloader.dataset import Basic_Dataset, SmokeCloud_Dataset_V2
from checkpoint.checkpoint import CheckpointMgr
from models.loss import FocalLoss_BCE
pj = os.path.join

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', default='/project/data/smoke_cloud/result/')
    parser.add_argument('--output',default='./pth/smoke_cloud_unet2d_smoke100k/')
    parser.add_argument('--lr',default=0.001,type=float)
    parser.add_argument('--max_epoch',default=100,type=int)
    parser.add_argument('--batchsize',default=8,type=int)
    parser.add_argument('--view_interval',default=50,type=int)
    parser.add_argument('--ckpt_interval',default=1000,type=int)
    parser.add_argument('--devnum',default=1,type=int)
    args = parser.parse_args()
    return args


def val(args):
    print('#'*10, 'EVAL' , '#'*10)
    if not os.path.exists(args.result):
        os.makedirs(args.result, exist_ok=True)
    # n_cuda_device = torch.cuda.device_count()
    n_cuda_device = min([args.devnum,4])
    # dataset_test = Basic_Dataset(datapath=args.datapath)
    dataset_test = SmokeCloud_Dataset_V2(ssspath=None, s100kpath='/project/data/smoke_cloud/smoke100k',mode='test')
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
    m_iou = 0
    m_t = 0
    mp_iou = 0
    mr_iou = 0
    m_fp_ratio = 0
    m_non_t = 0
    with torch.no_grad():
        for ind,batch in enumerate(dataloader):
            process_line = ind/len(dataloader)*100

            if m_t > 1:
                m_iou_score = m_iou/m_t
                mp_iou_score = mp_iou/m_t
                mr_iou_score = mr_iou/m_t
            else:
                m_iou_score = 0
                mp_iou_score = 0
                mr_iou_score = 0


            if m_non_t > 1:
                m_fp_score = m_fp_ratio / m_non_t
            else:
                m_fp_score = 0

            print('{:.2f}% done, with m_iou_score = {:.4f}, mriou = {:.4f}, mpiou = {:.4f}, m_fp = {:.4f}'\
                  .format(process_line, m_iou_score, mr_iou_score,  mp_iou_score, m_fp_score), end='\r')
            # if process_line > 2:
            #     break
            imgs = batch['X']#[b,c,h,w]
            ori_imgs = batch['img']#[b,h,w,c]
            gt_masks = batch['y']#[b,h,w]
            # print(gt_masks.shape)

            imgs = imgs.cuda()
            gt_masks = gt_masks.cuda()
            preds = model.inference(imgs)#[b,1,h,w]
            preds = preds.squeeze()

            _gt_masks = (gt_masks > 0.5) #[b,h,w]
            _preds = (preds > 0.5)

            joint = (_gt_masks&_preds).float()
            union = (_gt_masks|_preds).float()

            _non_obj_ = gt_masks.sum(dim=[1,2]) == 0#[b]
            _obj_ = ~_non_obj_

            _joint_obj_ = joint[_obj_]
            _union_obj_ = union[_obj_]
            _preds_obj_ = _preds[_obj_]
            _gt_masks_obj_ = _gt_masks[_obj_]


            if _joint_obj_.shape[0] > 0:

                _joint_obj_ = torch.sum(_joint_obj_, dim=[1,2]).cpu().data.numpy()
                _union_obj_ = torch.sum(_union_obj_, dim=[1, 2]).cpu().data.numpy()


                iou = _joint_obj_/(_union_obj_+1e-3) #[b]
                p_iou = _joint_obj_/ (torch.sum(_preds_obj_.float(),dim=[1,2]).cpu().data.numpy()+1e-3)
                r_iou = _joint_obj_/ (torch.sum(_gt_masks_obj_.float(),dim=[1,2]).cpu().data.numpy()+1e-3)

                m_t += _joint_obj_.shape[0]
                m_iou += iou.sum()
                mp_iou += p_iou.sum()
                mr_iou += r_iou.sum()

            _gt_masks_non_ = _gt_masks[_non_obj_]

            if _gt_masks_non_.shape[0] > 0:
                _preds_non_ = _preds[_non_obj_]
                fp_ratio = _preds_non_.sum(dim=[1,2]).cpu().data.numpy()/(224*224)
                m_fp_ratio += fp_ratio.sum()
                m_non_t += _gt_masks_non_.shape[0]




            # print(joint)
            # print(union)
            #
            # print(iou.sum(), iou)


            # break



            #
            # for img,pred, gt_mask in zip(ori_imgs, preds, gt_masks):
            #     img = img.cpu().data.numpy().astype(np.uint8) #[h,w,3] uint8
            #     pred = pred.cpu().data.numpy().squeeze() #[h,w] fp32
            #     gt_mask = gt_mask.cpu().data.numpy().squeeze() #uint8
            #
            #     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #     hsv[...,-1] = (hsv[...,-1]*pred).astype(np.uint8)
            #     img2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            #     img = np.concatenate([img,img2], axis=1)
            #
            #     # r_ch = img[...,-1]*1
            #     # # upred = pred*255
            #     # # r_ch[pred>0.1] = upred[pred>0.1]
            #     # r_ch[pred>0.3] = 255
            #     # img[...,-1] = r_ch.astype(np.uint8)
            #
            #     cv2.imwrite( pj(args.result,'{:04d}.png'.format(cnt)), img)
            #     cnt += 1


        print('#' * 10, 'EVAL END', '#' * 10)
        m_iou_score = m_iou/m_t
        m_fp_score = m_fp_ratio/m_non_t
        print('m_iou_score = {:.4f}, mriou = {:.4f}, mpiou = {:.4f}, mfp={:.4f}'.format(m_iou_score, mr_iou/m_t, mp_iou/m_t, m_fp_score))





if __name__ == '__main__':
    args = arg_parse()
    val(args)


