import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import numpy as np
import math, random
import qrcode
import utils.img_f as img_f

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

        self.final_size = config['final_size'] if 'final_size' in config else None
        

        random.seed(123)
        val_indexes = [random.randrange(0,10000) for i in range(200)]
        if split != 'train':
            self.indexes=val_indexes
            random.seed()
        else:
            self.indexes = list(range(10000))
            val_indexes.sort(reverse=True)
            for v in val_indexes:
                del self.indexes[v]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=2,
        )
        if self.indexes is not None:
            idx = self.indexes[idx]
        gt_char = '{}'.format(idx)
        qr.add_data(gt_char)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = np.array(img)
        if self.final_size is not None:
            img = img_f.resize(img,(self.final_size,self.final_size))
        img = (torch.from_numpy(img)[None,...].float()/255)*2 -1

        assert(img.max()==1 and img.min()==-1)

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
