from base import BaseModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
#import cv2
#TODO import models and discrimonators
from .discriminator import HybridDiscriminator
from .conv_gen import ConvGen
from .decoder_cnn import DecoderCNN
from skimage import draw
from scipy.ndimage.morphology import distance_transform_edt





class QRWraper(BaseModel):
    def __init__(self, config):
        super(QRWraper, self).__init__(config)

        style_dim = config['style_dim'] if 'style_dim' in config else 256
        self.style_dim = style_dim


        style_type = config['style'] if 'style' in config else 'normal'

        self.cond_disc=False
        self.vae=False


        #qr_type= config['qr_net'] #if 'qr_net' in config else 'default'
        #if 'DecoderCNN' in qr_type:
        #    self.qr_net=DecoderCNN(
        #elif 'none' in qr_type:
        #    self.qr_net=None
        #else:
        #    raise NotImplementedError('unknown QR model: '+qr_type)
        if 'pretrained_qr' in config and config['pretrained_qr'] is not None:
            snapshot = torch.load(config['pretrained_qr'], map_location='cpu')
            qr_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:4]=='qr_net.':
                    qr_state_dict[key[4:]] = value
            if len(qr_state_dict)==0:
                qr_state_dict=snapshot['state_dict']
                qr_config = snapshot['config']['model']
                config['qr_config'] = qr_config
            else:
                qr_config = config['qr_config']
            if 'arch' not in qr_config:
                qr_config['arch'] = 'DecoderCNN'
            self.qr_net = eval(qr_config['arch'])(qr_config)
            self.qr_net.load_state_dict( qr_state_dict )

        if 'generator' in config and config['generator'] == 'none':
            self.generator = None
        elif 'Conv' in config['generator']:
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            down_steps = config['down_steps'] if 'down_steps' in config else 3
            all_style = config['all_style'] if 'all_style' in config else False
            self.generator = ConvGen(style_dim,g_dim,down_steps,n_style_trans=n_style_trans,all_style=all_style)
        else:
            raise NotImplementedError('Unknown generator: {}'.format(config['generator']))

        if 'pretrained_generator' in config and config['pretrained_generator'] is not None:
            snapshot = torch.load(config['pretrained_generator'], map_location='cpu')
            gen_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:10]=='generator.':
                    gen_state_dict[key[10:]] = value
            self.generator.load_state_dict( gen_state_dict )

        if 'discriminator' in config and config['discriminator'] is not None:
            add_noise_img = config['disc_add_noise_img'] if 'disc_add_noise_img' in config else False
            if config['discriminator']=='down':
                self.discriminator = DownDiscriminator()
            elif 'simple' in config['discriminator']:
                self.discriminator=SimpleDisc()
            elif 'hybrid' in config['discriminator']:
                self.discriminator=HybridDiscriminator()
            elif config['discriminator']!='none':
                raise NotImplementedError('Unknown discriminator: {}'.format(config['discriminator']))

        if 'pretrained_discriminator' in config and config['pretrained_discriminator'] is not None:
            snapshot = torch.load(config['pretrained_discriminator'], map_location='cpu')
            discriminator_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:14]=='discriminator.':
                    discriminator_state_dict[key[14:]] = value
            self.discriminator.load_state_dict( discriminator_state_dict )


    def forward(self,qr_image,style=None):
        batch_size = qr_image.size(0)
        if style is None:
            style = torch.FloatTensor(batch_size,self.style_dim).normal_().to(qr_image.device)
        gen_img = self.generator(qr_image,style)
        return gen_img

