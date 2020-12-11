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

import sys
sys.path.append("./model")
sys.path.append("./datasets")
import qr_center_pixel_loss
import data_utils

import random, string
PADDING_CONSTANT = -1

def collate(batch):
    batch = [b for b in batch if b is not None]
    qr_img = torch.stack([b['image'] for b in batch],dim=0)

    d = {'qr_image': qr_img,
            'gt_char': [b['gt_char'] for b in batch],
            'targetchar': torch.stack([b['targetchar'] for b in batch],dim=0),
            'targetvalid': torch.cat([b['targetvalid'] for b in batch],dim=0),
            }

    if batch['masked_img'][0] is not None:
        d['masked_image'] = torch.stack([b['masked_img'] for b in batch], dim=0)
    return d

class SimpleQRDataset(Dataset):
    def __init__(self, dirPath,split,config):

        if "mask" in config and config["mask"]:
            qr = qr_center_pixel_loss.QRCenterPixelLoss(256, 33, 2, 0.1, bigger=True, split=False, factor=1, no_corners=False)
            self.mask = qr.get_mask(numpy=False)
        else:
            self.mask = None

        error_levels = {"l": 1, "m": 0, "q": 3, "h": 2}  # L < M < Q < H
        self.error_level = error_levels[config["error_level"]] if "error_level" in config else qrcode.constants.ERROR_CORRECT_L
        self.box_size = config["box_size"] if "box_size" in config else 1 # 6 in initial training
        self.border = config["border"] if "border" in config else 2 # 1 in initial training
        self.mask_pattern = config["mask_pattern"] if "mask_pattern" in config else None # 1 in initial training
        print(f"QR gen opts: {self.box_size} {self.border} {self.mask_pattern}")

        self.final_size = config['final_size'] if 'final_size' in config else None
        self.min_str_len = max(3,config["min_message_len"]) if 'min_message_len' in config else 4

        if "max_message_len" in config:
            config['str_len'] = config["max_message_len"]

        if 'total_random' in config or 'str_len' in config:
            self.str_len = config['total_random'] if 'total_random' in config else config['str_len']
            if "characters" not in config:
                if 'alphabet' in config and config['alphabet']=='digits':
                    self.characters = string.digits
                else:
                    self.characters = string.ascii_letters + string.digits + "-._~:/?#[]@!$&'()*+,;%" #url characters
            else:
                self.characters = config["characters"]
            self.char_to_index={char:n+1 for n,char in enumerate(self.characters)}
            self.char_to_index['\0']=0 #0 is actually reserved for predicting blank chars
        else:
            self.str_len = None
            self.char_to_index={'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'0':10,'\0':0}

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

        self.index_to_char={v:k for k,v in self.char_to_index.items()}

    def __len__(self):
        if self.str_len is None:
            return len(self.indexes)
        else:
            return 10000

    def __getitem__(self, idx):

        qr = qrcode.QRCode(
            version=1,
            error_correction=self.error_level,
            box_size=self.box_size,
            border=self.border,
            mask_pattern=self.mask_pattern,
        )
        if self.str_len is not None:
            #length = random.randrange(self.min_str_len,self.str_len+1)
            length = self.str_len
            gt_char = ''.join(random.choice(self.characters) for i in range(length))
        else:
            if self.indexes is not None:
                idx = self.indexes[idx]
            gt_char = '{}'.format(idx)
        qr.add_data(gt_char)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        img = np.array(img)
        if self.final_size is not None:
            img = img_f.resize(img,(self.final_size,self.final_size),degree=0)

        # Slight noise
        img = data_utils.gaussian_noise(img.astype(np.uint)*255, max_intensity=1)

        img = torch.from_numpy(img)[None,...].float()
        if img.max() == 255:
            img=img/255 #I think different versions of Pytorch do the conversion from bool to float differently
        img = img * 2 - 1

        targetchar = torch.LongTensor(self.str_len).fill_(0)
        for i,c in enumerate(gt_char):
            targetchar[i]=self.char_to_index[c]
        targetvalid = torch.FloatTensor([1])

        if not self.mask is None:
            masked_img = self.mask * img
        else:
            masked_img = None
        if False:
            x = img.squeeze().detach().numpy()
            import matplotlib.pyplot as plt
            plt.imshow((x+1)*127.5, cmap="gray");plt.show()
            plt.imshow((self.mask.permute(1,2,0)) * 255, cmap="gray"); plt.show()
            #image = data_utils.gaussian_noise(x*255, max_intensity=1)

        return {
            "image": img,
            "gt_char": gt_char,
            'targetchar': targetchar,
            'targetvalid': targetvalid,
            "masked_img": masked_img
        }

# add noise
# blur

if __name__=='__main__':
    SimpleQRDataset()