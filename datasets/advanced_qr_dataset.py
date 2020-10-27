import json
import cv2
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

            
class AdvancedQRDataset(Dataset):
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



    def save_dataset(self, save_path):
        pass

    def load_dataset(self, load_path):
        pass

    @staticmethod
    def apply_distortions(self,
                          image,
                          homography=True, #
                          blur=True,
                          superimpose=True):

        if homography:
            new_top_left     =
            new_top_right    =
            new_bottom_right =
            new_bottom_left  =
            h, status = cv2.findHomography(pts_src, pts_dst)
            im_out = cv2.warpPerspective(image, h, (im_dst.shape[1], im_dst.shape[0]))

    def generate_qr_code(self, gt_char):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=5,
            border=2,
            mask_pattern=1,
        )

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

    @staticmethod
    def create_message(self, l=15):
        return f"{np.random.randint(0,1e15)}:015d}"

    def __getitem__(self, idx):
        if self.indexes is not None:
            idx = self.indexes[idx]

        if self.data:
            return self.data[idx]
        else:
            return self.generate_new_qr_code()


def _make_char_set(all_char_string):
    """ Take in a huge string
            Get frequency of letters
            Create char_to_idx and idx_to_char dictionaries, indices based on alphabetical sort
    Args:
        all_char_string:

    Returns:

    """
    import collections
    char_freq = collections.Counter(all_char_string) # dict with {"letter":count, ...}
    od = collections.OrderedDict(sorted(char_freq.items()))
    idx_to_char, char_to_idx = {}, {}

    char_to_idx["|"] = 0
    idx_to_char[0] = "|"

    for i,key in enumerate(od.keys()):
        idx_to_char[i+1] = key
        char_to_idx[key] = i+1

    return char_to_idx, idx_to_char, char_freq

def blur(img, max_intensity=1.5):
    max_intensity = np.random.uniform(0, max_intensity)
    return ndimage.gaussian_filter(img, max_intensity)

def gaussian_noise(img, max_intensity=.1, logger=None):
    """
        Expects images on 0-255 scale
        max_intensity: .1 - light haze, 1 heavy

        Adds random noise to image
    """

    random_state = np.random.RandomState()
    sd = np.random.rand() * max_intensity / 2
    # min(abs(np.random.normal()) * max_intensity / 2, max_intensity / 2)
    #sd = max_intensity / 2  # ~95% of observations will be less extreme; if max_intensity=1, we set so 95% of multipliers are <1
    noise_mask = random_state.randn(*img.shape, ) * sd  # * 2 - max_intensity # min -occlusion, max occlusion
    noise_mask = np.clip(noise_mask, -1, 1) * 255/2
    noisy_img = np.clip(img + noise_mask, 0, 255)
    return noisy_img


if __name__=='__main__':
    pass
