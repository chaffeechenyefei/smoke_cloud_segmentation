from dataloader.augmentation_smoke_cloud import smoke100k_generator, smoke_semantic_segmentation
import cv2
import numpy as np
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', default=0,type=int)
    parser.add_argument('--i', default=100, type=int)
    parser.add_argument('--hw', default=224, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()

    img_idx = args.i

    genTor = smoke100k_generator(mode='train',mask_size=(args.hw,args.hw)) if args.v == 0 else smoke_semantic_segmentation(mask_size=(args.hw, args.hw))

    cv_img, mask_img = genTor.get_sample(idx=img_idx)

    mask_img = (mask_img*255).astype(np.uint8)
    print(cv_img.shape, mask_img.shape)

    cv2.imwrite('cv_img.jpg', cv_img)
    cv2.imwrite('mask_img.jpg', mask_img)