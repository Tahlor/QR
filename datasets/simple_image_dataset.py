import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision

from collections import defaultdict
import os
import numpy as np
import math, random
import utils.img_f as img_f

import random, string
from .simple_qr_dataset import SimpleQRDataset
PADDING_CONSTANT = -1


class SimpleImageDataset(Dataset):
    def __init__(self, dirPath,split,config):

        self.size = config['size'] if 'size' in config else None
        self.index=[]

        for root, subdirs, files in os.walk(dirPath):
            for filename in files:
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    self.index.append(os.path.join(root,filename))

        #print(self.index)

        self.transform =torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(0.01,0.01,0.01,0.1)
            ])

        self.qr_dataset=None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        img = None
        while img is None:
            img = img_f.imread(self.index[idx])
            idx = (img+7)%len(self.index)
        if img.shape[0]!=self.size or img.shape[1]!=self.size:
            img = img_f.resize(img,(self.size,self.size),degree=0)
        img = torch.from_numpy(img).permute(2,0,1).float()
        if img.max()>=220:
            img=img/255 #I think different versions of Pytorch do the conversion from bool to float differently
        if img.size(0)==4: #alpha channel, we'll fill it with a QR code image
            if self.qr_dataset is None:
                self.qr_dataset=SimpleQRDataset(None,'train',{'str_len':17,'final_size':self.size, 'noise':False})
            qr_img =self.qr_dataset[0]['image']
            qr_img = (qr_img+1)/2
            qr_img = qr_img.expand(3,-1,-1).clone() #convert to color
            qr_img[:,(img[3]>0).bool()] =img[:3,(img[3]>0).bool()] #add image where alpha is >0
            img=qr_img
        img = self.transform(img)
        img = img*2 -1

        return img,None
