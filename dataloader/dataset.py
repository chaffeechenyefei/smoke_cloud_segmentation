import torch
import os, cv2
import numpy as np
import random
from torch.utils.data import Dataset
from dataloader.augmentation import sample_generator
from dataloader.normalize import hwc_to_chw,normal_imagenet

pj = os.path.join


from dataloader.augmentation_smoke_cloud import basic_generator as basic_generator_smoker
class Basic_Dataset(Dataset):
    valid_ext = ['jpg','png','bmp','jpeg']
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath
        self.GenTor = basic_generator_smoker(datapath=datapath)
        self.len = self.GenTor.len()

    def __len__(self):
        return self.len


    def __getitem__(self, idx):
        cv_img, imgname = self.GenTor.get_sample(idx)
        img2 = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img2 = normal_imagenet(img2)
        img2 = hwc_to_chw(img2)
        t_img = torch.FloatTensor(img2.astype(np.float32))
        return {'X':t_img, 'img':cv_img}


"""
smoke cloud segmentation
"""
from dataloader.augmentation_smoke_cloud import smoke_cloud_generator
from dataloader.augmentation_smoke_cloud import smoke_semantic_segmentation
class SmokeCloud_Dataset(Dataset):
    def __init__(self, bgpath, pngpath, ssspath=None ,mode='train', iter_times:int = 1e5):
        super().__init__()
        self.bgpath = bgpath
        self.pngpath = pngpath
        self.sampler = [smoke_cloud_generator(bgRep=bgpath, pngRep=pngpath)]
        if ssspath is not None:
            self.sampler.append(smoke_semantic_segmentation(datapath=ssspath))
        self.mode = mode
        self.iter_times = iter_times

    def __len__(self):
        return self.iter_times

    def __getitem__(self, idx):
        use_rand = True if self.mode =='train' else False

        sampler = random.choice(self.sampler)
        img, mask = sampler.get_sample(idx, use_rand)

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = normal_imagenet(img2)
        img2 = hwc_to_chw(img2)

        img2 = torch.FloatTensor(img2.astype(np.float32))
        mask = torch.FloatTensor(mask.astype(np.float32))
        if self.mode == 'train':
            return {'X':img2, 'y':mask}
        else:
            img = torch.FloatTensor(img.astype(np.float32)) #[h,w,c]
            return {'X': img2, 'y': mask, 'img':img}


from dataloader.augmentation_smoke_cloud import smoke100k_generator
class SmokeCloud_Dataset_V2(Dataset):
    def __init__(self,  ssspath, s100kpath ,mode='train', mask_size=(224,224) ,iter_times:int = 1e5):
        super().__init__()
        self.sampler = [smoke100k_generator(datapath=s100kpath, mode=mode, mask_size=mask_size)]
        if ssspath is not None:
            self.sampler.append(smoke_semantic_segmentation(datapath=ssspath, mask_size=mask_size))
        self.mode = mode
        self.iter_times = int(iter_times)
        self.mask_size = mask_size

    def __len__(self):
        if self.mode == 'train':
            return self.iter_times
        else:
            return self.sampler[0].len()

    def __getitem__(self, idx):
        use_rand = True if self.mode =='train' else False

        sampler = random.choice(self.sampler)
        img, mask = sampler.get_sample(idx, use_rand)

        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img2 = normal_imagenet(img2)
        img2 = hwc_to_chw(img2)

        img2 = torch.FloatTensor(img2.astype(np.float32))
        mask = torch.FloatTensor(mask.astype(np.float32))
        if self.mode == 'train':
            return {'X':img2, 'y':mask}
        else:
            img = torch.FloatTensor(img.astype(np.float32)) #[h,w,c]
            return {'X': img2, 'y': mask, 'img':img}



# """
# 高空抛物
# """
# class VirtualMOD_Dataset(Dataset):
#     def __init__(self, bg_path, fg_path=None, ext='.bmp',T=16):
#         super(VirtualMOD_Dataset,self).__init__()
#         self.bg_path = bg_path
#         self.fg_path = fg_path
#         self.ext = ext
#         self.T = T
#         self.sampler = sample_generator(bg_path=bg_path,fg_path=fg_path,ext=ext)
#
#     def __len__(self):
#         return self.sampler.len()
#
#     def __getitem__(self, idx):
#         if random.random() < 0.5:
#             fg_idx = -1
#         else:
#             fg_idx = random.randint(0,self.sampler.len()-1)
#         imgs,mask = self.sampler.get_sample(bg_idx=idx, fg_idx=fg_idx,T=self.T)
#         imgs = normal_imagenet(imgs)
#         imgs = hwc_to_chw(imgs)
#
#         assert len(mask.shape) == 2, 'mask shape {}'.format(mask.shape)
#
#         imgs = [ torch.FloatTensor(im.astype(np.float32)) for im in imgs ]#[ [c,h,w] ]
#         mask = torch.FloatTensor(mask.astype(np.float32))
#
#         img = torch.stack(imgs,dim=1) #[c,t,h,w]
#         return {'X':img,'y':mask}
#
# """
# 火焰0-1分类
# """
# from dataloader.augmentation_dunnings import dunnings_fire_sample_generator, fire_dataset_sample_generator
# class DunningsFire_Dataset(Dataset):
#     def __init__(self, datapath, bg_path=None , bg_ratio=0.1 ,mode='train', use_additional_data=False):
#         super(DunningsFire_Dataset, self).__init__()
#         self.bgratio = bg_ratio
#         self.bg_path = bg_path
#         self.datapath = datapath
#         self.sampler_dunnings = dunnings_fire_sample_generator(datapath=datapath, bgpath=bg_path, bgratio=self.bgratio ,mode=mode)
#         if use_additional_data:
#             # additional_path = '/project/data/Fire/fire-dataset/{}'.format('train' if mode == 'train' else 'validation')
#             additional_path = '/project/data/Fire/fire-dataset/{}'.format('train')
#             self.sampler_fire_dataset = fire_dataset_sample_generator(datapath= additional_path, mode=mode)
#         else:
#             self.sampler_fire_dataset = None
#
#     def __len__(self):
#         return self.sampler_dunnings.len() + (0 if self.sampler_fire_dataset is None else self.sampler_fire_dataset.len())
#
#
#     def __getitem__(self, idx):
#         if idx >= self.sampler_dunnings.len() and self.sampler_fire_dataset is not None:
#             idx = idx - self.sampler_dunnings.len()
#             img, label = self.sampler_fire_dataset.get_sample(idx)
#         else:
#             img, label = self.sampler_dunnings.get_sample(idx)
#         #bgr->rgb
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = normal_imagenet(img)
#         img = hwc_to_chw(img)
#
#         img = torch.FloatTensor(img.astype(np.float32))
#         return {'X':img, 'y':label}
#
# """
# water segmentation
# """
# from dataloader.augmentation_kaggle_water import kaggle_water_sample_generator
# class KaggleWater_Dataset(Dataset):
#     def __init__(self, datapath, mode='train', expand_ratio:int=100):
#         super().__init__()
#         self.datapath = datapath
#         self.sampler = kaggle_water_sample_generator(datapath=datapath, mode=mode)
#         self.mode = mode
#         self.expand_ratio = expand_ratio
#
#     def __len__(self):
#         if self.mode == 'train':
#             return self.expand_ratio*self.sampler.len()
#         else:
#             return self.sampler.len()
#
#     def __getitem__(self, idx):
#         img, mask = self.sampler.get_sample(idx)
#
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img2 = normal_imagenet(img2)
#         img2 = hwc_to_chw(img2)
#
#         img2 = torch.FloatTensor(img2.astype(np.float32))
#         mask = torch.FloatTensor(mask.astype(np.float32))
#         if self.mode == 'train':
#             return {'X':img2, 'y':mask}
#         else:
#             img = torch.FloatTensor(img.astype(np.float32)) #[h,w,c]
#             return {'X': img2, 'y': mask, 'img':img}
#
# from dataloader.augmentation_kaggle_water import uavdt_car_background_generator, iccv_water_puddle_generator, bdd_car_background_generator, itsc_flood_generator
# class KaggleWater_ext_Dataset(Dataset):
#     def __init__(self, datapath, uavdatapath, iccvdatapath , bdddatapath, itscdatapath,mode='train', expand_ratio:list=[100,1,10,1,100]):
#         super().__init__()
#         self.datapath = datapath
#         self.sampler = kaggle_water_sample_generator(datapath=datapath, mode=mode)
#         self.uavdt_sampler = None
#         self.iccv_sampler = None
#         self.bdd_sampler = None
#         self.itsc_sampler = None
#         if mode == 'train':
#             self.uavdt_sampler = uavdt_car_background_generator(datapath=uavdatapath)
#             self.iccv_sampler = iccv_water_puddle_generator(datapath=iccvdatapath)
#             self.bdd_sampler = bdd_car_background_generator(datapath=bdddatapath)
#             self.itsc_sampler = itsc_flood_generator(datapath=itscdatapath)
#
#         self.mode = mode
#         self.er0 = int(expand_ratio[0])
#         self.er1 = int(expand_ratio[1])
#         self.er2 = int(expand_ratio[2])
#         self.er3 = int(expand_ratio[3])
#         self.er4 = int(expand_ratio[4])
#
#     def __len__(self):
#         if self.mode == 'train':
#             return self.er0*self.sampler.len() + self.er1*self.uavdt_sampler.len() + \
#                    self.er2*self.iccv_sampler.len() + self.er3*self.bdd_sampler.len() + \
#                    self.er4*self.itsc_sampler.len()
#         else:
#             return self.sampler.len()
#
#     def __getitem__(self, idx):
#         if self.mode == 'train':
#             if idx < self.er0*self.sampler.len():
#                 img, mask = self.sampler.get_sample(idx)
#             elif idx < self.er0*self.sampler.len() + self.er1*self.uavdt_sampler.len():
#                 _idx = idx - self.er0*self.sampler.len()
#                 img, mask = self.uavdt_sampler.get_sample(_idx)
#             elif idx < self.er0*self.sampler.len() + self.er1*self.uavdt_sampler.len() + self.er2*self.iccv_sampler.len():
#                 _idx = idx - self.er0*self.sampler.len() - self.er1*self.uavdt_sampler.len()
#                 img, mask = self.iccv_sampler.get_sample(_idx)
#             elif idx < self.er0*self.sampler.len() + self.er1*self.uavdt_sampler.len() + self.er2*self.iccv_sampler.len()+ self.er3*self.bdd_sampler.len():
#                 _idx = idx - self.er0*self.sampler.len() - self.er1*self.uavdt_sampler.len() - self.er2*self.iccv_sampler.len()
#                 img, mask = self.bdd_sampler.get_sample(_idx)
#             else:
#                 _idx = idx - self.er0*self.sampler.len() - self.er1*self.uavdt_sampler.len() - self.er2*self.iccv_sampler.len() - self.er3*self.bdd_sampler.len()
#                 img, mask = self.itsc_sampler.get_sample(_idx)
#         else:
#             img, mask = self.sampler.get_sample(idx)
#
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img2 = normal_imagenet(img2)
#         img2 = hwc_to_chw(img2)
#
#         img2 = torch.FloatTensor(img2.astype(np.float32))
#         mask = torch.FloatTensor(mask.astype(np.float32))
#         if self.mode == 'train':
#             return {'X':img2, 'y':mask}
#         else:
#             img = torch.FloatTensor(img.astype(np.float32)) #[h,w,c]
#             return {'X': img2, 'y': mask, 'img':img}
#
#
# from dataloader.augmentation_kaggle_water import basic_generator
# class WaterBasicDataset(Dataset):
#     def __init__(self, datapath):
#         super().__init__()
#         self.datapath = datapath
#         self.sampler = basic_generator(datapath=datapath)
#
#     def __len__(self):
#         return self.sampler.len()
#
#     def __getitem__(self, idx):
#
#         img, imgname = self.sampler.get_sample(idx)
#
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img2 = normal_imagenet(img2)
#         img2 = hwc_to_chw(img2)
#
#         img2 = torch.FloatTensor(img2.astype(np.float32))
#         img = torch.FloatTensor(img.astype(np.float32)) #[h,w,c]
#         return {'X': img2, 'name': imgname , 'img':img}