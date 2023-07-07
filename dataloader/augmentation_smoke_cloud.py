import sys
import os
import cv2
import numpy as np
import random
pj = os.path.join
# sys.path.append("./")
# sys.path.append("../")
from dataloader.pipeline.aug import Pipeline as General_Pipeline



"""
basic_generator
"""
class basic_generator(object):
    valid_ext = ['jpg','png','bmp','jpeg']
    def __init__(self, datapath):
        super(basic_generator, self).__init__()
        self.datapath = datapath

        self.imgnames = []
        self.imgnames = [pj(self.datapath, c) for c in os.listdir(self.datapath) if c.split('.')[-1].lower() in self.valid_ext ]
        self.img_num = len(self.imgnames)

        print('-> basic_generator:: total {:d} images found from {}' \
              .format(self.img_num, datapath))

    def len(self):
        return self.img_num

    def get_sample(self, idx: int):
        idx = idx % self.img_num
        """
        normal data
        """
        imgname = self.imgnames[idx]

        cv_img = cv2.imread(imgname)
        cv_img = cv2.resize(cv_img, dsize=(224, 224))


        return cv_img, os.path.basename(imgname)


"""
from smoke100k
"""
class smoke100k_generator(object):
    imgext = '.png'
    def __init__(self, datapath = '/project/data/smoke_cloud/smoke100k/', mode='train', mask_size=(224,224)):
        super().__init__()
        self.datapath = datapath
        self.mode = 'train' if mode == 'train' else 'test'
        self.Limgpath = pj(datapath, 'smoke100k-L/{}'.format(self.mode))
        self.Mimgpath = pj(datapath, 'smoke100k-M/{}'.format(self.mode))
        self.Himgpath = pj(datapath, 'smoke100k-H/{}'.format(self.mode))

        self.imgpath = [self.Limgpath, self.Mimgpath , self.Himgpath ]

        self.subpath_mask = 'smoke_mask'
        self.subpath_img = 'smoke_image'
        self.subpath_bg = 'smoke_free_image'

        self.maskimgnames = []
        self.bgimgnames = []

        self.mask_size = mask_size

        for imgpath in self.imgpath:
            """
            imgpath = /project/data/smoke_cloud/smoke100k/smoke100k-{L/M/H}/train
            
            self.imgnames = /project/data/smoke_cloud/smoke100k/smoke100k-{L/M/H}/train/smoke_mask/*.png
            self.bgimgnames = /project/data/smoke_cloud/smoke100k/smoke100k-{L/M/H}/train/smoke_free_image/*.png
            """
            self.maskimgnames.extend( [ pj(imgpath, self.subpath_mask ,c) for c in os.listdir( pj(imgpath, self.subpath_mask)) if c.endswith(self.imgext)] )
            self.bgimgnames.extend( [ pj(imgpath, self.subpath_bg, c) for c in os.listdir(pj(imgpath, self.subpath_bg)) if c.endswith(self.imgext)] )

        self.num_imgs = len(self.maskimgnames)
        self.num_bgimgs = len(self.bgimgnames)

        print('-> smoke100k_generator:: total {:d} images and {:d} bg images found'.format(self.num_imgs, self.num_bgimgs))

        self.GenTor = General_Pipeline(dst_size=(224, 224), rotation=[-45, 45], crop=[0.7, 1.0])

    def len(self):
        return int(self.num_imgs+self.num_bgimgs)

    def get_sample(self, idx:int, use_rand=False):
        if not use_rand:
            idx = idx % (self.num_imgs+self.num_bgimgs)
            if idx < self.num_imgs:
                maskimgname = self.maskimgnames[idx]
                ##ATT: imgpath should not contain self.subpath_mask
                imgname = maskimgname.replace(self.subpath_mask, self.subpath_img)
            else:
                imgname = self.bgimgnames[idx-self.num_imgs]
                maskimgname = None
        else:
            if random.random() < 0.2:
                maskimgname = random.choice(self.maskimgnames)
                imgname = maskimgname.replace(self.subpath_mask, self.subpath_img)
            else:
                imgname = random.choice(self.bgimgnames)
                maskimgname = None

        cv_img = cv2.imread(imgname)
        if maskimgname is None:
            mask_img = np.zeros( cv_img.shape[:2], np.uint8)
        else:
            mask_img = cv2.imread(maskimgname, cv2.IMREAD_GRAYSCALE )
            mask_img = (mask_img > 10).astype(np.uint8)*255

        if self.mode == 'train':
            cv_img, mask_img =  self.GenTor(cv_img, mask_img)
            mask_img = cv2.resize(mask_img, self.mask_size)
        else:
            cv_img = cv2.resize(cv_img,(224,224))
            mask_img = cv2.resize(mask_img,self.mask_size)

        mask_img = mask_img.astype(np.float32)/255
        return cv_img, mask_img

"""
from git/Smoke-semantic-segmentation
"""
class smoke_semantic_segmentation(object):
    imgext = '.jpg'
    def __init__(self, datapath = '/project/data/smoke_cloud/smoke_images_with_annot/tagged/', mask_size=(224,224)):
        super().__init__()
        self.datapath = datapath
        self.imgpath = pj(datapath,'images')
        self.maskpath = pj(datapath, 'masks')
        # image和mask同名
        self.imgnames = [c for c in os.listdir(self.imgpath) if c.endswith(self.imgext)]
        self.num_imgs = len(self.imgnames)

        print('-> smoke_semantic_segmentation:: total {:d} images found'.format(self.num_imgs))

        self.GenTor = General_Pipeline(dst_size=(224,224), rotation=[-45,45], crop=[0.7,1.0])
        self.mask_size = mask_size

    def len(self):
        return self.num_imgs

    def get_sample(self, idx:int, use_rand=False):
        if not use_rand:
            idx = idx % self.num_imgs
            imgname = self.imgnames[idx]
        else:
            imgname = random.choice(self.imgnames)
        cv_img = cv2.imread(pj(self.imgpath, imgname))
        mask_img = cv2.imread(pj(self.maskpath, imgname), cv2.IMREAD_GRAYSCALE )

        cv_img, mask_img =  self.GenTor(cv_img, mask_img)
        mask_img = cv2.resize(mask_img, self.mask_size)
        mask_img = mask_img.astype(np.float32)/255
        return cv_img, mask_img


"""
smoke_cloud_dataset
"""
class smoke_cloud_generator(object):
    imgext = '.jpg'
    def __init__(self, bgRep:str , pngRep:str):
        super().__init__()
        self.bgpath = bgRep
        self.pngpath = pngRep

        self.bgimgnames = [pj(self.bgpath, c) for c in os.listdir(self.bgpath) if c.endswith(self.imgext)]
        self.bgimg_num = len(self.bgimgnames)

        print('-> smoke_cloud_generator:: total {:d} bg images found' \
              .format(self.bgimg_num))

        self.GenTor = _smoke_cloud_generator_(pngRep=pngRep, png_scale=(0.2,0.6))
        self.BGGenTor = BG_Pipeline(dst_size=(224,224))

    def len(self):
        return self.bgimg_num


    def get_sample(self, idx:int, use_rand=False):
        if not use_rand:
            idx = idx%self.bgimg_num
            bgimgname = self.bgimgnames[idx]
        else:
            bgimgname = random.choice(self.bgimgnames)
        """bg"""
        bg_img = cv2.imread(bgimgname)
        bg_img = self.BGGenTor(bg_img)

        """add target"""
        target_img, mask_img = self.GenTor.gen(bg_img, dirty_num=3, alpha=random.uniform(0.7,1.0))

        mask_img = mask_img.astype(np.float32)/255
        return target_img, mask_img







"""
png贴图工具
"""
class _smoke_cloud_generator_(object):
    valid_ext = ['png']

    def __init__(self, pngRep='/project/data/dirty_data/dirty_png/dirty_png/', png_scale=(0.2,0.6)):
        super().__init__()
        print('-> dirty_generator init')
        self.png_scale = png_scale
        print('reading png')
        imgnames = [c for c in os.listdir(pngRep) if c.split('.')[-1] in self.valid_ext]
        self.dirty_imgs = []
        for imgname in imgnames:
            # print(imgname)
            img_cv = cv2.imread(pj(pngRep, imgname), cv2.IMREAD_UNCHANGED)
            self.dirty_imgs.append(img_cv)
        self.pipeline1 = Pipeline([-180, 180], [0.85, 1.0])
        self.pipeline2 = Pipeline([-180, 180], [0.3, 0.5])
        print('<- dirty_generator init')

    def gen(self, imgInp, dirty_num=5, alpha=0.8, selectNum=None):
        """
        1. 在原图生成贴图位置
        2. 从self.dirt_imgs中随机抽取进行贴图：
            2.1 crop部分进行贴图
            2.2 全图进行贴图
            2.3 旋转等操作
        """
        H, W = imgInp.shape[:2]
        bboxes = self.gen_box(H, W, dirty_num, scale=self.png_scale)

        patches, patch_masks = [], []

        for i in range(dirty_num):
            # if selectNum:
            #     di = selectNum
            # else:
            #     di = random.choice(list(range(len(self.dirty_imgs))))
            # dirty_img = self.dirty_imgs[di]
            # print('self.dirty_imgs[{:d}]'.format(di))
            dirty_img = random.choice(self.dirty_imgs)
            dirty_img, dirty_mask = self.crop_to_box(dirty_img, bboxes[i])
            patches.append(dirty_img)
            patch_masks.append(dirty_mask)

        return self.gen_dirty_on_whole(imgInp, bboxes, patches, patch_masks, alpha)

    def gen_box(self, h, w, num=5, scale=(0.1, 0.3)):
        """
        在原图尺寸上生成num个box，每个box的比例在scale范围内
        h,w是原图的尺寸，scale是生成box长宽与原图的比例
        """
        xs = [int(w * random.random()) for c in range(num)]
        ys = [int(h * random.random()) for c in range(num)]
        ws = [int(w * (random.random() * (scale[1] - scale[0]) + scale[0])) for c in range(num)]
        hs = [int(h * (random.random() * (scale[1] - scale[0]) + scale[0])) for c in range(num)]
        bboxes = []
        for i in range(num):
            _x, _y, _w, _h = xs[i], ys[i], ws[i], hs[i]
            _w = min([_w, w - _x])
            _h = min([_h, h - _y])
            box = [_x, _y, _w, _h]
            bboxes.append(box)
        return bboxes

    def crop_to_box(self, imgInp, box):
        """
        输入dirty的png矢量图
        将输入的图像按照box的大小进行裁剪，同时进行一定的图像增强
        """
        w, h = box[2:]
        img = imgInp[:, :, :3]
        mask = imgInp[:, :, 3]
        if random.random() < 0.5:
            img, mask = self.pipeline1(img, (w, h), mask)
        else:
            img, mask = self.pipeline2(img, (w, h), mask)
        return img, mask

    def gen_dirty_on_whole(self, imgInp, bboxes, patches, patch_masks, alpha):
        """
        将矢量图贴到整个图像上
        """
        H, W = imgInp.shape[:2]
        blkImg = np.zeros((H, W, 3), np.float32)
        blkMsk = np.zeros((H, W), np.float32)
        binMsk = np.zeros((H, W), np.float32)
        for box, patch, patch_mask in zip(bboxes, patches, patch_masks):
            m = patch_mask < 2
            patch[m, :] = 0
            _x, _y, _w, _h = box
            blkMsk[_y:_y + _h, _x:_x + _w] += patch_mask
            blkImg[_y:_y + _h, _x:_x + _w] += patch  # 存在一些问题，背景的val 0不应该参与进来
            binMsk[_y:_y + _h, _x:_x + _w] += (patch_mask > 0)
        binMsk[binMsk < 1] = 1
        blkMsk /= binMsk
        blkImg /= binMsk[:, :, np.newaxis]

        alphaMtx = blkMsk / 255
        alphaMtx = alphaMtx[:, :, np.newaxis] * alpha
        imgMerge = alphaMtx * blkImg + (1 - alphaMtx) * imgInp

        return imgMerge.astype(np.uint8), blkMsk.astype(np.uint8)


"""
图像处理pipeline
"""
class RandomCrop(object):
    def __init__(self, crop_ratio=[0.85, 0.95]):
        self.crop_ratio = crop_ratio
        if crop_ratio is not None:
            assert min(crop_ratio) > 0 and max(crop_ratio) <= 1

    def __call__(self, img_cv2, mask=None, dst_sz=None):
        h, w = img_cv2.shape[0:2]
        ratio = np.random.uniform(low=min(self.crop_ratio), high=max(self.crop_ratio))
        new_h, new_w = int(h * ratio), int(w * ratio)
        x0 = np.random.randint(low=0, high=w - new_w)
        y0 = np.random.randint(low=0, high=h - new_h)
        img_cv2 = img_cv2[y0: y0 + new_h, x0:x0 + new_w]
        if dst_sz is not None:
            img_cv2 = cv2.resize(img_cv2, dst_sz)
        else:
            img_cv2 = cv2.resize(img_cv2, (w, h))
        if mask is not None:
            mask = mask[y0: y0 + new_h, x0:x0 + new_w]
            if dst_sz is not None:
                mask = cv2.resize(mask, dst_sz)
            else:
                mask = cv2.resize(mask, (w, h))
        return img_cv2, mask

    def __repr__(self):
        return self.__class__.__name__ + '(crop_ratio={})'.format(
            self.crop_ratio)


class RandomFlip(object):
    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, img_cv2, mask=None):
        flip = True if np.random.rand() < self.flip_ratio else False
        if flip:
            if self.direction == 'horizontal':
                img_cv2 = np.flip(img_cv2, axis=1)  # 左右镜像
                if mask is not None:
                    mask = np.flip(mask, axis=1)
            else:
                img_cv2 = np.flip(img_cv2, axis=0)  # 上下镜像
                if mask is not None:
                    mask = np.flip(mask, axis=0)

        return img_cv2, mask

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(self.flip_ratio)


class Rotate(object):
    def __init__(self, angle_range=[-10, 10]):
        self.angle_range = angle_range

    def _rotate(self, img_cv2, angle):
        if img_cv2 is not None:
            h, w = img_cv2.shape[0:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)
            img_res = cv2.warpAffine(img_cv2, M, (w, h))
            return img_res
        else:
            return None

    def __call__(self, img_cv2, mask=None):
        angle = random.uniform(a=min(self.angle_range), b=max(self.angle_range))
        return self._rotate(img_cv2, angle), self._rotate(mask, angle)


class Pipeline(object):
    def __init__(self, rotation=[-30, 30], crop=[0.85, 1.0], **kwargs):
        super(Pipeline, self).__init__()
        self.rotate_func = Rotate(angle_range=rotation)
        self.random_crop = RandomCrop(crop_ratio=crop)
        self.random_flip = RandomFlip(flip_ratio=0.5, direction='horizontal')

    def __call__(self, img_cv2, dst_size=(112, 112), mask=None):
        if np.random.rand() < 0.5:
            img_cv2, mask = self.random_crop(img_cv2, mask=mask)
        if np.random.rand() < 0.7:
            img_cv2, mask = self.rotate_func(img_cv2, mask=mask)
        img_cv2 = cv2.resize(img_cv2, dsize=dst_size)
        img_cv2, mask = self.random_flip(img_cv2, mask)
        if mask is not None:
            mask = cv2.resize(mask, dsize=dst_size)
        return img_cv2, mask

"""
BG_Pipeline
"""
class BG_Pipeline(object):
    def __init__(self, dst_size=(224,224)):
        super(BG_Pipeline, self).__init__()
        self.random_flip_h = RandomFlip(flip_ratio=0.5, direction='horizontal')
        self.random_flip_v = RandomFlip(flip_ratio=0.5, direction='vertical')
        self.random_crop = RandomCrop(crop_ratio=[0.5, 1.0])
        self.dst_size = dst_size

    def __call__(self, img_cv2):
        img_cv2, _ = self.random_flip_h(img_cv2)
        img_cv2, _ = self.random_flip_v(img_cv2)
        img_cv2, _ = self.random_crop(img_cv2)
        img_cv2 = cv2.resize(img_cv2, self.dst_size)
        return img_cv2





