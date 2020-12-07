import re
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
from .style_gan import GrowGen, GrowDisc
from .style_gan2 import SG2Generator, SG2Discriminator, SG2DiscriminatorPatch
from .grow_gen_u import GrowGenU
from .style_gan2_u_gen import SG2UGen



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
        elif 'qr_net' in config:
            qr_config = config['qr_net']
            if 'arch' not in qr_config:
                qr_config['arch'] = 'DecoderCNN'
            self.qr_net = eval(qr_config['arch'])(qr_config)
        else:
            self.qr_net = None

        if 'generator' in config and config['generator'] == 'none':
            self.generator = None
        elif 'Conv' in config['generator']:
            g_dim = config['gen_dim'] if 'gen_dim' in config else 256
            n_style_trans = config['n_style_trans'] if 'n_style_trans' in config else 6
            down_steps = config['down_steps'] if 'down_steps' in config else 3
            all_style = config['all_style'] if 'all_style' in config else False
            skip_connections = not config['no_gen_skip'] if 'no_gen_skip' in config else True
            diff_gen = not config['no_diff_gen'] if 'no_diff_gen' in config else True
            self.generator = ConvGen(style_dim,g_dim,down_steps,n_style_trans=n_style_trans,all_style=all_style,skip_connections=skip_connections,diff_gen=diff_gen)
        elif 'GrowGenU' in config['generator']:
            self.generator = GrowGenU(style_dim)
        elif 'Grow' in config['generator']:
            self.generator = GrowGen(style_dim)
        elif 'SG2UGen' in config['generator']:
            if 'small' in config['generator']:
                channel_multiplier=1
            else:
                channel_multiplier=2
            coord_conv = 'oord' in config['generator']
            use_tanh = 'unbound' not in config['generator']
            predict_offset = 'predict_offset' in config['generator']
            self.generator = SG2UGen(256,style_dim,8,channel_multiplier=channel_multiplier,coord_conv=coord_conv,use_tanh=use_tanh)
        elif 'StyleGAN2' in config['generator']:
            self.generator = SG2Generator(256,style_dim,8,channel_multiplier=2)
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
                use_minibatch_stat = config['disc_use_minibatch_stat'] if 'disc_use_minibatch_stat' in config else False
                self.discriminator=HybridDiscriminator(use_minibatch_stat=use_minibatch_stat)
            elif 'Grow' in config['discriminator']:
                self.discriminator=GrowDisc()
            elif 'StyleGAN2' in config['discriminator']:
                if 'small' in config['discriminator']:
                    channel_multiplier=1
                else:
                    channel_multiplier=2
                smaller='smaller' in config['discriminator']
                if "patch" in config['discriminator'].lower():
                    receptive_field_mask = "receptive_field" in config['discriminator'].lower()
                    corner_mask = "corner" in config['discriminator'].lower()
                    conv_layers = re.findall("(layers)([0-9]+)", config['discriminator'].lower())
                    conv_layers = int(conv_layers[0][1]) if conv_layers else None
                    print("Conv layers: ", conv_layers)
                    self.discriminator = SG2DiscriminatorPatch(256,
                                                               channel_multiplier=channel_multiplier,
                                                               smaller=smaller,
                                                               qr_size=21,
                                                               padding=2,
                                                               receptive_field_mask=receptive_field_mask,
                                                               corner_mask=corner_mask,
                                                               conv_layers=conv_layers)
                else:
                    mask_corners = config['discriminator'] if 'mask_corners' in config['discriminator'] else False
                    qr_size = config['qr_size'] if 'qr_size' in config else 21
                    qr_padding = config['qr_padding'] if 'qr_padding' in config else 2
                    self.discriminator=SG2Discriminator(256,
                                                        channel_multiplier=channel_multiplier,
                                                        smaller=smaller,
                                                        mask_corners=mask_corners,
                                                        qr_size=qr_size,
                                                        padding=qr_padding)
            elif config['discriminator']!='none':
                raise NotImplementedError('Unknown discriminator: {}'.format(config['discriminator']))

        if 'pretrained_discriminator' in config and config['pretrained_discriminator'] is not None:
            snapshot = torch.load(config['pretrained_discriminator'], map_location='cpu')
            discriminator_state_dict={}
            for key,value in  snapshot['state_dict'].items():
                if key[:14]=='discriminator.':
                    discriminator_state_dict[key[14:]] = value
            self.discriminator.load_state_dict( discriminator_state_dict )


    def forward(self,qr_image,style=None,step=None, alpha=None,return_latent=False):
        batch_size = qr_image.size(0)
        if style is None:
            style = [torch.FloatTensor(batch_size,self.style_dim).normal_().to(qr_image.device)]
        if step is not None:
            gen_img = self.generator(qr_image,style,step,alpha)
            return gen_img
        else:
            return self.generator(qr_image,style,return_latents=return_latent)
