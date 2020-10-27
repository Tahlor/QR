import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import numpy as np
import math, random
import qrcode

import random
PADDING_CONSTANT = -1

def collate(batch):
    batch = [b for b in batch if b is not None]
    return {'image': torch.stack([b['image'] for b in batch],dim=0),
            'gt_char': [b['gt_char'] for b in batch],
            'targetchar': torch.stack([b['targetchar'] for b in batch],dim=0),
            'targetvalid': torch.cat([b['targetvalid'] for b in batch],dim=0)
            }
            

class SimpleQRDataset(Dataset):
    def __init__(self, dirPath,split,config):

        self.char_to_index={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':10,'\0':0}
        self.index_to_char={v:k for k,v in self.char_to_index.items()}


        if split != 'train':
            random.seed(123)
            self.indexes = [random.randrange(0,10000) for i in range(200)]
            random.seed()
        else:
            self.indexes = None

    def __len__(self):
        return 10000 if self.indexes is None else len(self.indexes)

    def __getitem__(self, idx):

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=5,
            border=2,
        )

        if self.indexes is not None:
            idx = self.indexes[idx]
        gt_char = '{}'.format(idx)
        qr.add_data(gt_char)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = (torch.from_numpy(np.array(img))[None,...].float())*2 -1

        targetchar = torch.LongTensor(17).fill_(0)
        for i,c in enumerate(gt_char):
            targetchar[i]=self.char_to_index[c]
        targetvalid = torch.FloatTensor([1])

        return {
            "image": img,
            "gt_char": gt_char,
            'targetchar': targetchar,
            'targetvalid': targetvalid
        }
