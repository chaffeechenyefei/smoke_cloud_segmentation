from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from prefetch_generator import BackgroundGenerator
class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(),max_prefetch=2)#chaffee

class zDataLoader(object):

    def __init__(self,
                 imgs_per_gpu,
                 workers_per_gpu,
                 num_gpus=1,
                 dist=False,
                 shuffle=True,
                 pin_memory=True,
                 use_original = False,
                 verbose = True,
                 **kwargs):
        super(zDataLoader, self).__init__()
        print('################### Init Dataloader. ###################') if verbose else None
        self.imgs_per_gpu = imgs_per_gpu
        self.workers_per_gpu = workers_per_gpu
        self.num_gpus = num_gpus
        self.dist = dist
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.use_original = use_original

        self.batch_size = imgs_per_gpu * num_gpus
        self.num_workers = workers_per_gpu * num_gpus

    def __call__(self, dataset):
        if not self.dist:
            if self.use_original:
                return DataLoader(dataset=dataset,
                                   batch_size=self.batch_size,
                                   shuffle=self.shuffle,
                                   num_workers=self.num_workers,
                                   pin_memory=self.pin_memory,
                                   drop_last=False)
            else:
                return DataLoaderX(dataset=dataset,
                                   batch_size=self.batch_size,
                                   shuffle=self.shuffle,
                                   num_workers=self.num_workers,
                                   pin_memory=self.pin_memory,
                                   drop_last=False)
        else:
            return DataLoader(dataset=dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              sampler=DistributedSampler(dataset),
                              num_workers=self.num_workers,
                              pin_memory=self.pin_memory,
                              drop_last=False)