from time import sleep
import json
from easydict import EasyDict as edict
import cv2
import string
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import numpy as np
import math, random
import qrcode
from datasets import utils
import random
from matplotlib import pyplot as plt
from pathlib import Path

PADDING_CONSTANT = -1

## TODO:
#### Save/load datasets
# Get homogrpahy to work
# Superimpose with lots of images
# OCCLUDE
# Check if QR code reader can read QR codes after distortions
# Support more arguments from config

def collate(batch):
    batch = [b for b in batch if b is not None]
    return {'image': torch.stack([b['image'] for b in batch],dim=0),
            'gt_char': [b['gt_char'] for b in batch],
            'targetchar': torch.stack([b['targetchar'] for b in batch],dim=0),
            'targetvalid': torch.cat([b['targetvalid'] for b in batch],dim=0)
            }

class AdvancedQRDataset(Dataset):
    def __init__(self, dirPath,split,config, full_config, *args, **kwargs):
        self.full_config = full_config = edict(full_config)
        self.character_set = config.alphabet
        self.char_to_index, self.index_to_char = _make_char_set(config.alphabet)
        self.data = None
        self.qr_size = config.image_size #full_config.model.input_size
        self.max_message_len = config.max_message_len
        self.images = list(Path(config.background_image_path).glob("*"))
        if config.distortions:
            self.distortions = {"homography":False,
                                "blur":True,
                                "superimpose":False,
                                "add_noise":True,
                                "distortion":True,
                                "rotate":False,
                                "occlude":False,
                                "background_image":None}
        else:
            self.distortions = False

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
    def apply_distortions(image,
                          homography=True, #
                          blur=True,
                          superimpose=True,
                          add_noise=True,
                          rotate=True,
                          distortion=True,
                          occlude=True,
                          background_image=None):
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
        if image.ndim != 3:
            image = image[:, :, np.newaxis]

        if homography and False: # DOESN'T WORK
            image = utils.homography(image)

        if rotate:
            raise NotImplemented

        if occlude:
            raise NotImplemented

        if distortion:
            image = utils.elastic_transform(image)

        if add_noise:
            image = utils.gaussian_noise(image)

        if blur:
            image = utils.blur(image)

        if superimpose:
            if background_image is None:
                background_image = AdvancedQRDataset.get_random_image()
            image = utils.superimpose_images(image, background_image)

        return image

    def get_random_image(self):
        #filename = "../dev/landscape.png"
        return cv2.imread(filename=random.choice(self.images))

    def generate_qr_code(self, gt_char):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=6,
            border=1,
            mask_pattern=1,
        )
        gt_char = str(gt_char)
        qr.add_data(gt_char)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black",
                            back_color="white").resize(self.qr_size)
        #print(np.array(img).shape)
        img = np.array(img)

        targetchar = torch.LongTensor(17).fill_(0)
        for i,c in enumerate(gt_char):
            targetchar[i]=self.char_to_index[c]
        targetvalid = torch.FloatTensor([1])
        return {
            "image": utils.img2tensor(img),
            "gt_char": gt_char,
            'targetchar': targetchar,
            'targetvalid': targetvalid,
            "image_undistorted": img.astype(float) * 255
        }

    def create_message(self, l=15):
        return ''.join(random.choices(self.character_set, k=l))

    def __getitem__(self, idx):
        if self.indexes is not None:
            idx = self.indexes[idx]

        if self.data:
            img_dict = self.data[idx]
        else:
            img_dict = self.generate_qr_code(self.create_message(random.randint(self.max_message_len,self.max_message_len)))

        if self.distortions:
            #img1 = img_dict["image"].clone()
            img = AdvancedQRDataset.apply_distortions(img_dict["image_undistorted"], **self.distortions)
            img_dict["image"] = img2 = utils.img2tensor(img.squeeze()) # DOES NOT HANDLE 3 channel color
            if False:
                i1, i2 = img.squeeze(), img_dict["image_undistorted"].squeeze()
                i = np.c_[i1, i2]
                plt.hist(i1.flatten())
                plt.show()
                plt.hist(i2.flatten())
                plt.show()
                plt.imshow(i, cmap="gray");
                plt.show()
            # plt.hist(img2.numpy().flatten()); plt.show()
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

def test_distortions():
    img = cv2.imread(filename = "../dev/landscape.png")
    img = cv2.resize(img, (img.shape[0], 270))
    #plt.imshow(img); plt.show()
    # cv2.imshow("image1", img.copy())
    # cv2.waitKey()

    x = AdvancedQRDataset.apply_distortions(img,
                                            homography=False,
                                            blur=False,
                                            superimpose=False,
                                            add_noise=True,
                                            distortion=False,
                                            rotate=False,
                                            occlude=False,
                                            background_image=None)
    plt.imshow(x); plt.show()
    assert np.allclose(x, img)


if __name__=='__main__':
    test_distortions()
    cv2.destroyAllWindows()
