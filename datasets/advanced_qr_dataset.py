from time import sleep
import json
from easydict import EasyDict as edict
import cv2
import string
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import torchvision
from collections import defaultdict
import os
import numpy as np
import math, random
import qrcode
from datasets import data_utils
import random
from matplotlib import pyplot as plt
from pathlib import Path
import sys
sys.path.append("..")
from utils import util

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

class AdvancedQRDataset(Dataset):
    def __init__(self, dirPath,split,config, full_config, *args, **kwargs):
        self.full_config = full_config = edict(full_config)
        self.character_set = config.alphabet
        self.char_to_index, self.index_to_char = _make_char_set(config.alphabet)
        self.data = None
        self.qr_size = config.image_size #full_config.model.input_size
        self.final_size = full_config.model.input_size
        self.resize_op = torchvision.transforms.Resize(self.final_size)
        self.coordconv = config.coordconv


        self.max_message_len = config.max_message_len
        if "background_image_path" in config:
            path = (Path(config.background_image_path) / "files.json")
            if path.exists():
                self.images = json.load(path.open())
            else:
                self.images = [x.as_posix() for x in Path(config.background_image_path).rglob("*.JPEG")]
                json.dump(self.images, path.open("w"))
        else:
            self.images = None

        if "distortions" in kwargs:
            self.distortions = kwargs["distortions"]
        elif config.distortions:
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
                          background_images=None,
                          distortion_prob=.7):
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

        if superimpose and background_images and np.random.random()<distortion_prob:
            background_image = AdvancedQRDataset.get_random_image(background_images)
            image = data_utils.superimpose_images(image, background_image) #[:,:,np.newaxis]

        if homography and False: # DOESN'T WORK
            image = data_utils.homography(image)

        if rotate:
            raise NotImplemented

        if occlude and np.random.random()<distortion_prob:
            image = data_utils.occlude(image)

        if distortion and np.random.random()<distortion_prob:
            image = data_utils.elastic_transform(image)

        if add_noise and np.random.random()<distortion_prob:
            image = data_utils.gaussian_noise(image)

        if blur and np.random.random()<distortion_prob:
            image = data_utils.blur(image)

        return image

    @staticmethod
    def get_random_image(images):
        while True:
            try:
                img = random.choice(images) if images else "../dev/landscape.png"
                img = cv2.imread(filename=img, flags=cv2.IMREAD_GRAYSCALE)[:, :, np.newaxis]
                if not img is None:
                    break
            except:
                pass
        return img

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
        img = np.array(img).astype(np.uint8) * 255

        targetchar = torch.LongTensor(17).fill_(0)
        for i,c in enumerate(gt_char):
            targetchar[i]=self.char_to_index[c]
        targetvalid = torch.FloatTensor([1])
        return {
            "gt_char": gt_char,
            'targetchar': targetchar,
            'targetvalid': targetvalid,
            "image_undistorted": img
        }

    def is_valid_qr(self, image, message=""):
        """

        Args:
            image: W,H,C, 0-255
            message:

        Returns:

        """
        return not data_utils.qr_decode(image) is None

    # def set_coordconv(self, type_as=torch.DoubleTensor):
    #     x_dim, y_dim = self.qr_size
    #     xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
    #     yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)
    #
    #     xx_channel = xx_channel.float() / (x_dim - 1)
    #     yy_channel = yy_channel.float() / (y_dim - 1)
    #
    #     xx_channel = (xx_channel * 2 - 1).type(type_as)
    #     yy_channel = (yy_channel * 2 - 1).type(type_as)
    #
    #     self.xx_channel = xx_channel.transpose(1, 2).float()
    #     self.yy_channel = yy_channel.transpose(1, 2).float()
    # def coordconv(self, input_tensor):
    #
    #     return torch.cat([
    #         input_tensor,
    #         self.xx_channel,
    #         self.yy_channel], dim=0)
    def add_coordconv(self, input_tensor):
        x_dim, y_dim = self.qr_size
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = (xx_channel * 2 - 1)
        yy_channel = (yy_channel * 2 - 1)

        xx_channel = xx_channel.transpose(1, 2).float()
        yy_channel = yy_channel.transpose(1, 2).float()

        return torch.cat([
            input_tensor,
            xx_channel,
            yy_channel], dim=0)

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
            img = AdvancedQRDataset.apply_distortions(img_dict["image_undistorted"].copy(), **self.distortions)
            img_dict["image"] = img2 = FACTOR(data_utils.img2tensor(img.squeeze())) # DOES NOT HANDLE 3 channel color
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
            img_dict["targetvalid"] = torch.FloatTensor([1 if self.is_valid_qr(img) else 0])
            #print(img_dict["targetvalid"])
        else:
            img_dict["image"] = FACTOR(data_utils.img2tensor(img_dict["image_undistorted"]))

        if self.qr_size[0] != self.final_size[0]:
            img_dict["image"] = self.resize_op(img_dict["image"])

        # Add coordconv
        if self.coordconv:
            img_dict["image"] = self.add_coordconv(img_dict["image"])
        # print(img_dict["image"])
        # stop

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

def test_distortions(dataset=AdvancedQRDataset, image="../dev/landscape.png"):
    img = cv2.imread(filename = image) if isinstance(image, str) else image
    img = cv2.resize(img, (img.shape[0], 270))
    #plt.imshow(img); plt.show()
    # cv2.imshow("image1", img.copy())
    # cv2.waitKey()

    x = dataset.apply_distortions(img,
                                homography=False,
                                blur=True,
                                superimpose=True,
                                add_noise=True,
                                distortion=False,
                                rotate=False,
                                occlude=True,
                                background_images=None)
    plt.imshow(x); plt.show()
    assert np.allclose(x, img)


def test_dataset():
    config = "../configs/___distortions.conf"
    config = edict(json.load(Path(config).open()))
    config = data_utils.make_config_consistent(config)
    images = json.load((Path(config.data_loader.background_image_path) / "files.json").open())
    distortions = {"homography":False,
                                "blur":False,
                                "superimpose":True,
                                "add_noise":False,
                                "distortion":False,
                                "rotate":False,
                                "occlude":True,
                                "background_images": images}


    dataset = AdvancedQRDataset(dirPath=None,
                                split="train",
                                config=config.data_loader,
                                full_config=config,
                                distortions=distortions)
    dataset[0]
    #test_distortions(dataset, image=dataset[0]["image_undistorted"].squeeze())

def test():
    test_distortions()

def test_loader():
    import data_loaders
    config = "../configs/___distortions.conf"
    config = edict(json.load(Path(config).open()))
    config = data_utils.make_config_consistent(config)
    x = data_loaders.getDataLoader(config,"train")
    iter(x)

if __name__=='__main__':
    if True:
        test_dataset()
        cv2.destroyAllWindows()
    else:
        test()