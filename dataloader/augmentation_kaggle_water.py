"""
iccv_water_puddle_generator
kaggle_water_sample_generator
uavdt_car_background_generator
bdd_car_background_generator
basic_generator
itsc_flood_generator
"""
import os
import cv2
import random
import numpy as np

from dataloader.pipeline.aug import Pipeline

pj = os.path.join


"""
basic_generator
"""
class basic_generator(object):
    imgext = '.jpg'
    def __init__(self, datapath):
        super(basic_generator, self).__init__()
        self.datapath = datapath

        self.imgnames = []
        self.imgnames = [pj(self.datapath, c) for c in os.listdir(self.datapath) if c.endswith(self.imgext)]
        self.img_num = len(self.imgnames)

        print('-> basic_generator:: total {:d} images found' \
              .format(self.img_num))

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
bdd100k
"""
class bdd_car_background_generator(object):
    imgext = '.jpg'

    def __init__(self, datapath, mode='train'):
        super(bdd_car_background_generator, self).__init__()
        assert mode in ['train'], 'mode should be in [train]'
        self.mode = mode
        self.datapath = datapath
        foldnames = ['10k/train', '100k/train']

        self.imgnames = []
        for fdname in foldnames:
            img_fd_full = pj(self.datapath, fdname)
            if os.path.isdir(img_fd_full):
                imgnames = [pj(img_fd_full, c) for c in os.listdir(img_fd_full) if c.endswith(self.imgext)]
                self.imgnames.extend(imgnames)

        self.pipeline = Pipeline(dst_size=(224, 224), rotation=[-45, 45], crop=[0.5, 0.99])
        self.img_num = len(self.imgnames)

        print('-> bdd_car_background_generator:: total {:d} images found' \
              .format(self.img_num))

    def len(self):
        return self.img_num

    def get_sample(self, idx: int):
        idx = idx % self.img_num
        """
        normal data
        """
        imgname = self.imgnames[idx]

        cv_img = cv2.imread(imgname)
        cv_label = np.zeros((224, 224), np.uint8)

        if cv_img is None:
            print(imgname)

        if self.mode == 'train':
            cv_img, _ = self.pipeline(cv_img, None)
        else:
            cv_img = cv2.resize(cv_img, dsize=(224, 224))

        cv_label = cv_label.astype(np.float32) / 255

        return cv_img, cv_label


"""
uavdt_car_background_generator
"""
class uavdt_car_background_generator(object):
    imgext = '.jpg'
    def __init__(self, datapath, mode='train'):
        super(uavdt_car_background_generator, self).__init__()
        assert mode in ['train'], 'mode should be in [train]'
        self.mode = mode
        self.datapath = datapath
        exfoldnames = ['M0201','M0208','M0209','M0301','M0401','M0402','M0403','M0501','M1301','M1304','M1305','M1306']
        foldnames = [c for c in os.listdir(self.datapath) if os.path.isdir(pj(self.datapath, c)) ]
        foldnames = [c for c in foldnames if c not in exfoldnames]

        self.imgnames = []
        for fdname in foldnames:
            img_fd_full = pj(self.datapath, fdname)
            if os.path.isdir(img_fd_full):
                imgnames = [pj(img_fd_full,c) for c in os.listdir(img_fd_full) if c.endswith(self.imgext)]
                self.imgnames.extend(imgnames)

        self.pipeline = Pipeline(dst_size=(224, 224), rotation=[-45, 45], crop=[0.5, 0.99])
        self.img_num = len(self.imgnames)

        print('-> uavdt_car_background_generator:: total {:d} images found' \
              .format(self.img_num))

    def len(self):
        return self.img_num

    def get_sample(self, idx:int):
        idx = idx%self.img_num
        """
        normal data
        """
        imgname = self.imgnames[idx]

        cv_img = cv2.imread(imgname)
        cv_label = np.zeros((224,224), np.uint8)

        if cv_img is None:
            print(imgname)

        if self.mode == 'train':
            cv_img, _ = self.pipeline(cv_img, None)
        else:
            cv_img = cv2.resize(cv_img, dsize=(224,224))

        cv_label = cv_label.astype(np.float32) / 255

        return cv_img, cv_label


"""
itsc flood dataset
images/image_{}.jpg
labels/label_{}.png
"""
class itsc_flood_generator(object):
    imgext = '.jpg'
    annoext = '.png'
    def __init__(self, datapath, mode='train'):
        super(itsc_flood_generator, self).__init__()
        assert mode in ['train'], 'mode should be in [train]'
        self.mode = mode
        self.datapath = datapath
        self.imgpath = pj(datapath,'images')
        self.annopath = pj(datapath,'labels')

        self.imgnames = []
        self.annonames = []

        imgnames_noext = [ (c.split(self.imgext)[0]).replace('image_','') for c in os.listdir(self.imgpath) if c.endswith(self.imgext)]
        annonames_noext = [ (c.split(self.annoext)[0]).replace('label_','') for c in os.listdir(self.annopath) if c.endswith(self.annoext)]
        imgnames_noext = list( set(imgnames_noext)&set(annonames_noext) )
        imgnames = [ pj(self.imgpath, 'image_'+c+self.imgext) for c in imgnames_noext ]
        annonames = [ pj(self.annopath, 'label_'+c+self.annoext) for c in imgnames_noext ]

        self.imgnames.extend(imgnames)
        self.annonames.extend(annonames)

        self.pipeline = Pipeline(dst_size=(224, 224), rotation=[-45,45], crop=[0.5,0.99])

        self.img_num = len(self.imgnames)
        self.anno_num = len(self.annonames)

        print('-> itsc_flood_generator:: total {:d} images and {:d} anno images found'\
              .format(self.img_num, self.anno_num))
        assert self.img_num == self.anno_num, 'image num != anno num'

    def len(self):
        return self.img_num

    def get_sample(self, idx:int):
        idx = idx%self.img_num
        """
        normal data
        """
        imgname = self.imgnames[idx]
        annoname = self.annonames[idx]


        cv_img = cv2.imread(imgname)
        cv_label = cv2.imread(annoname)

        if cv_label is None:
            print(annoname)
        if cv_img is None:
            print(imgname)

        if len(cv_label.shape) == 3:
            if cv_label.shape[-1] == 4:
                cv_label = cv2.cvtColor(cv_label, cv2.COLOR_BGRA2GRAY)
            else:
                cv_label = cv2.cvtColor(cv_label, cv2.COLOR_BGR2GRAY)

        cv_label = np.uint8(cv_label > 0)*255

        if self.mode == 'train':
            cv_img, cv_label = self.pipeline(cv_img, cv_label)
        else:
            cv_img = cv2.resize(cv_img, dsize=(224,224))
            cv_label = cv2.resize(cv_label, dsize=(224,224))

        cv_label = cv_label.astype(np.float32) / 255

        return cv_img, cv_label



"""
iccv_water_puddle_generator
"""
class iccv_water_puddle_generator(object):
    imgext = '.png'
    annoext = '.png'
    def __init__(self, datapath, mode='train'):
        super(iccv_water_puddle_generator, self).__init__()
        assert mode in ['train'], 'mode should be in [train]'
        self.mode = mode
        self.datapath = datapath
        self.imgpath = pj(datapath,'images')
        self.annopath = pj(datapath,'masks')
        foldnames = ['off_road','on_road']

        print('total folder:', foldnames)

        self.imgnames = []
        self.annonames = []
        for fdname in foldnames:
            img_fd_full = pj(self.imgpath, fdname)
            anno_fd_full = pj(self.annopath, fdname)
            if os.path.isdir(img_fd_full):
                imgnames_noext = [ (c.split(self.imgext)[0]).replace('img_','') for c in os.listdir(img_fd_full) if c.endswith(self.imgext)]
                annonames_noext = [ (c.split(self.annoext)[0]).replace('left_mask_','') for c in os.listdir(anno_fd_full) if c.endswith(self.annoext)]
                imgnames_noext = list( set(imgnames_noext)&set(annonames_noext) )
                imgnames = [ pj(img_fd_full, 'img_'+c+self.imgext) for c in imgnames_noext ]
                annonames = [ pj(anno_fd_full, 'left_mask_'+c+self.annoext) for c in imgnames_noext ]

                self.imgnames.extend(imgnames)
                self.annonames.extend(annonames)

        self.pipeline = Pipeline(dst_size=(224, 224), rotation=[-45,45], crop=[0.5,0.99])

        self.img_num = len(self.imgnames)
        self.anno_num = len(self.annonames)

        print('-> iccv_water_puddle_generator:: total {:d} images and {:d} anno images found'\
              .format(self.img_num, self.anno_num))
        assert self.img_num == self.anno_num, 'image num != anno num'

    def len(self):
        return self.img_num

    def get_sample(self, idx:int):
        idx = idx%self.img_num
        """
        normal data
        """
        imgname = self.imgnames[idx]
        annoname = self.annonames[idx]


        cv_img = cv2.imread(imgname)
        cv_label = cv2.imread(annoname)

        if cv_label is None:
            print(annoname)
        if cv_img is None:
            print(imgname)

        if len(cv_label.shape) == 3:
            if cv_label.shape[-1] == 4:
                cv_label = cv2.cvtColor(cv_label, cv2.COLOR_BGRA2GRAY)
            else:
                cv_label = cv2.cvtColor(cv_label, cv2.COLOR_BGR2GRAY)

        cv_label = np.uint8(cv_label > 0)*255

        if self.mode == 'train':
            cv_img, cv_label = self.pipeline(cv_img, cv_label)
        else:
            cv_img = cv2.resize(cv_img, dsize=(224,224))
            cv_label = cv2.resize(cv_label, dsize=(224,224))

        cv_label = cv_label.astype(np.float32) / 255

        return cv_img, cv_label

"""
kaggle_water_sample_generator
"""
class kaggle_water_sample_generator(object):
    imgext = '.jpg'
    annoext = '.png'
    def __init__(self, datapath, mode='train'):
        super(kaggle_water_sample_generator, self).__init__()
        assert mode in ['train','test'], 'mode should be in [train,test]'
        self.mode = mode
        self.datapath = datapath
        self.imgpath = pj(datapath,'JPEGImages')
        self.annopath = pj(datapath,'Annotations')
        splitfile = 'train.txt' if mode == 'train' else 'val.txt'
        splitfile = pj(datapath, splitfile)
        fp = open(splitfile, 'r')
        foldnames = fp.readlines()
        foldnames = [c.replace('\n','') for c in foldnames if c != '\n']

        print('total folder:', foldnames)

        self.imgnames = []
        self.annonames = []
        for fdname in foldnames:
            img_fd_full = pj(self.imgpath, fdname)
            anno_fd_full = pj(self.annopath, fdname)
            if os.path.isdir(img_fd_full):
                imgnames_noext = [ c.split(self.imgext)[0] for c in os.listdir(img_fd_full) if c.endswith(self.imgext)]
                annonames_noext = [ c.split(self.annoext)[0] for c in os.listdir(anno_fd_full) if c.endswith(self.annoext)]
                imgnames_noext = list( set(imgnames_noext)&set(annonames_noext) )
                imgnames = [ pj(img_fd_full, c+self.imgext) for c in imgnames_noext ]
                annonames = [ pj(anno_fd_full, c+self.annoext) for c in imgnames_noext ]

                self.imgnames.extend(imgnames)
                self.annonames.extend(annonames)

        self.pipeline = Pipeline(dst_size=(224, 224), rotation=[-45,45], crop=[0.5,0.99])

        self.img_num = len(self.imgnames)
        self.anno_num = len(self.annonames)

        print('-> kaggle_water_sample_generator:: total {:d} images and {:d} anno images found'\
              .format(self.img_num, self.anno_num))
        assert self.img_num == self.anno_num, 'image num != anno num'

    def len(self):
        return self.img_num

    def get_sample(self, idx:int):
        idx = idx%self.img_num
        """
        normal data
        """
        imgname = self.imgnames[idx]
        annoname = self.annonames[idx]


        cv_img = cv2.imread(imgname)
        cv_label = cv2.imread(annoname)

        if cv_label is None:
            print(annoname)
        if cv_img is None:
            print(imgname)

        if len(cv_label.shape) == 3:
            if cv_label.shape[-1] == 4:
                cv_label = cv2.cvtColor(cv_label, cv2.COLOR_BGRA2GRAY)
            else:
                cv_label = cv2.cvtColor(cv_label, cv2.COLOR_BGR2GRAY)

        cv_label = np.uint8(cv_label > 0)*255

        if self.mode == 'train':
            cv_img, cv_label = self.pipeline(cv_img, cv_label)
        else:
            cv_img = cv2.resize(cv_img, dsize=(224,224))
            cv_label = cv2.resize(cv_label, dsize=(224,224))

        cv_label = cv_label.astype(np.float32) / 255

        return cv_img, cv_label