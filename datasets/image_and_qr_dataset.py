import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from collections import defaultdict
import os
import numpy as np
import math, random
import torchvision

import utils.img_f as img_f
from .simple_qr_dataset import SimpleQRDataset
from .advanced_qr_dataset import AdvancedQRDataset
from .advanced_qr_dataset3 import AdvancedQRDataset3

from .simple_image_dataset import SimpleImageDataset

def collate(batch):
    batch = [b for b in batch if b is not None]
    d = {
            'image': torch.stack([b['image'] for b in batch],dim=0),
            'qr_image': torch.stack([b['qr_image'] for b in batch],dim=0),
            'gt_char': [b['gt_char'] for b in batch],
            'targetchar': torch.stack([b['targetchar'] for b in batch],dim=0),
            'targetvalid': torch.cat([b['targetvalid'] for b in batch],dim=0)
            }
    if 'masked_img' in batch[0] and batch[0]['masked_img'] is not None:
        d['masked_image'] = torch.stack([b['masked_img'] for b in batch], dim=0)
    return d

class ImageAndQRDataset(Dataset):
    def __init__(self, dirPath,split,config):
        
        if config['QR_dataset']['data_set_name']=='SimpleQRDataset':
            self.qr_dataset = SimpleQRDataset('none',split,config['QR_dataset'])
        elif config['QR_dataset']['data_set_name']=='AdvancedQRDataset':
            self.qr_dataset = AdvancedQRDataset('none',split,config['QR_dataset'])
        elif config['QR_dataset']['data_set_name']=='AdvancedQRDataset3':
            self.qr_dataset = AdvancedQRDataset3('none',split,config['QR_dataset'])
        else:
            raise NotImplementedError('Unknown QR dataset: {}'.format(config['QR_dataset']['data_set_name']))

        if config['image_dataset_name']=='LSUN':
            transform =torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop([256,256]),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ColorJitter(0.1,0.1,0.1,0.1)
                    ])
            if split=='valid':
                split='val'
            self.image_dataset = torchvision.datasets.LSUN(dirPath,classes=['{}_{}'.format(config['image_class'],split)],transform=transform)
        elif config['image_dataset_name']=='simple':
            self.image_dataset = SimpleImageDataset(dirPath,None,config['image_dataset_config'])


    def __len__(self):
        return max(len(self.qr_dataset),len(self.image_dataset))

    def __getitem__(self, idx):
        
        qr_index = random.randrange(len(self.qr_dataset))       #idx%len(self.qr_dataset)
        image_index = random.randrange(len(self.image_dataset)) #idx//len(self.qr_dataset)

        qr_data = self.qr_dataset[qr_index]
        image,target = self.image_dataset[image_index]
        
        if type(image) is not torch.Tensor:
            image = np.array(image)
            image = (2*(torch.from_numpy(image).float()/255)-1).permute(2,0,1)

        qr_image = qr_data['image']
        #if qr_image.size(1)!=image.size(1) or qr_image.size(2)!=image.size(2):
        #    #qr_image = img_f.resize(qr_image[0],image.shape[1:])[None,...]
        #    qr_image = F.interpolate(qr_image[None,...],image.shape[1:],mode='bilinear')[0]
        assert(qr_image.size(1)==image.size(1))

        qr_data['qr_image'] = qr_image
        qr_data['image'] = image
        return qr_data
