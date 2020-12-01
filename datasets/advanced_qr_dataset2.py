from time import sleep
import json
#from easydict import EasyDict as edict
import string
import torch
import torchvision
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import numpy as np
import math, random
import qrcode
from datasets import brian_data_utils as data_utils
import random
from matplotlib import pyplot as plt
from pathlib import Path
import sys
sys.path.append("..")
from utils import util
from utils import img_f

PADDING_CONSTANT = -1

## TODO:
#### Save/load datasets
# Get homogrpahy to work
# OCCLUDE

FACTOR = lambda x : x

def collate(batch):
    batch = [b for b in batch if b is not None]
    return {'image': torch.stack([b['image'] for b in batch],dim=0),
            'gt_char': [b['gt_char'] for b in batch],
            'targetchar': torch.stack([b['targetchar'] for b in batch], dim=0),
            'targetvalid': torch.cat([b['targetvalid'] for b in batch], dim=0)
            }

class AdvancedQRDataset2(Dataset):
    def __init__(self, dirPath,split,config, *args, **kwargs):
        if config['alphabet']=='digits':
            self.character_set = string.digits
        elif config['alphabet']=='url':
            self.character_set = string.ascii_letters + string.digits + "-._~:/?#[]@!$&'()*+,;%" #url characters
        self.char_to_index, self.index_to_char = _make_char_set(self.character_set)
        self.data = None

        self.final_size = config['final_size'] if 'final_size' in config else 256
        self.border = 2 if 'qr_border' not in config else config['qr_border']
        self.mask_pattern =None if 'qr_mask_pattern' not in config else config['qr_mask_pattern']

        self.max_message_len = config['max_message_len'] if 'max_message_len' in config else 17
        self.min_message_len = config['min_message_len'] if 'min_message_len' in config else 4
        if "background_image_path" in config:
            if 'use_lsun' in config and config['use_lsun']:
                self.images = torchvision.datasets.LSUN(config['background_image_path'],classes=['bedroom_train'])
            else:
                path = (Path(config['background_image_path']) / "files.json")
                if path.exists():
                    self.images = json.load(path.open())
                else:
                    self.images = [x.as_posix() for x in Path(config['background_image_path']).rglob("*.JPEG")]
                    self.images += [x.as_posix() for x in Path(config['background_image_path']).rglob("*.jpg")]
                    self.images += [x.as_posix() for x in Path(config['background_image_path']).rglob("*.jpeg")]
                    self.images += [x.as_posix() for x in Path(config['background_image_path']).rglob("*.png")]
                    json.dump(self.images, path.open("w"))
        else:
            self.images = None

        if "distortions" in kwargs:
            self.distortions = kwargs["distortions"]
        elif config['distortions']:
            self.distortions = {"homography":False,
                                "blur":True,
                                "superimpose":True,
                                "add_noise":True,
                                "distortion":True,
                                "rotate":False,
                                "occlude":True,
                                "background_images":self.images}
            #self.occlude = data_utils.Occlude()
        else:
            self.distortions = False


    def __len__(self):
        return 10000

    def save_dataset(self, save_path):
        pass

    def load_dataset(self, load_path):
        pass


    @staticmethod
    def apply_distortions(image,
                          homography=True, #
                          blur=True,
                          superimpose=True,
                          add_noise=True,
                          rotate=True,
                          distortion=True,
                          occlude=True,
                          background_images=None):
        """


        Args:
            image:
            homography:
            blur:
            superimpose:
            add_noise (bool): add (RGB) contrast noise
            rotate: 90/180/270 degree rotation - maybe don't use this at first
            distortion: spatially distort image
            occlude:
            background_image:

        Returns:

        """
        # if image.ndim != 3:
        #     image = image[:, :, np.newaxis]
        if superimpose and background_images and np.random.random()>0.2:#.3:
            background_image = AdvancedQRDataset2.get_random_image(background_images)
            image = img_f.superimposeImages(image, background_image) #[:,:,np.newaxis]

        #if homography and False: # DOESN'T WORK
        #    image = data_utils.homography(image)

        if rotate:
            raise NotImplemented

        if occlude and np.random.random()>.3:
            image = data_utils.occlude(image)

        if distortion and np.random.random()>.4:
            image = data_utils.elastic_transform(image)

        if add_noise and np.random.random()>.4:
            image = data_utils.gaussian_noise(image)

        if blur and np.random.random()>.4:
            image = data_utils.blur(image)

        return image

    @staticmethod
    def get_random_image(images):
        img = random.choice(images) if images else "../dev/landscape.png"
        if type(img) is str:
            return img_f.imread(img)
        else:
            return np.array(img[0])

    def generate_qr_code(self, gt_char):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=self.border,
            mask_pattern=self.mask_pattern,
        )
        gt_char = str(gt_char)
        qr.add_data(gt_char)
        qr.make(fit=True) #WARNING
        img = qr.make_image(fill_color="black",
                            back_color="white")#.resize(self.qr_size)
        #print(np.array(img).shape)
        img = np.array(img).astype(np.uint8) * 255
        img = img_f.resize(img,(self.final_size,self.final_size),degree=0)
        img = img[:,:,None].repeat(3,axis=2)


        return {
            "gt_char": gt_char,
            "image_undistorted": img
        }



    def create_message(self, l=15):
        return ''.join(random.choices(self.character_set, k=l))

    def __getitem__(self, idx):
        if self.data:
            img_dict = self.data[idx]
        else:
            img_dict = self.generate_qr_code(self.create_message(random.randint(self.min_message_len,self.max_message_len)))

        if self.distortions:
            #img1 = img_dict["image"].clone()
            img = AdvancedQRDataset2.apply_distortions(img_dict["image_undistorted"].copy(), **self.distortions)
            gt_char = util.zbar_decode(img)
            #assert(gt_char is None or type(gt_char) is str)
            if type(gt_char) is not str:
                gt_char=None
        else:
            img = img_dict["image_undistorted"]
            gt_char = img_dict['gt_char']
        
        img = torch.from_numpy(img).float().permute(2,0,1)
        img=img/255
        img = img*2 -1
        img_dict["image"] = img
        targetchar = torch.LongTensor(self.max_message_len).fill_(0)
        if gt_char is not None:
            for i,c in enumerate(gt_char):
                try:
                    targetchar[i]=self.char_to_index[c]
                except KeyError:
                    pass
            targetvalid = torch.FloatTensor([1])
        else:
            targetvalid = torch.FloatTensor([0])
        img_dict['targetchar'] = targetchar
        img_dict['targetvalid'] = targetvalid
        img_dict['gt_char'] = gt_char

        return img_dict

def _make_char_set(all_char_string):
    """ Take in a huge string
            Create char_to_idx and idx_to_char dictionaries
    Args:
        all_char_string:

    Returns:

    """
    idx_to_char, char_to_idx = {}, {}
    blank ='\0'
    char_to_idx[blank] = 0
    idx_to_char[0] = blank
    assert blank not in all_char_string # this will mess up the alphabet size
    chars = [char for char in all_char_string];
    chars.sort()

    for i,key in enumerate(chars):
        idx_to_char[i+1] = key
        char_to_idx[key] = i+1

    return char_to_idx, idx_to_char

