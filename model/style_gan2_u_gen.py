import torch
from torch import nn
import torch.nn.functional as F
from .style_gan2 import PixelNorm,EqualLinear,ConstantInput,StyledConv,ToRGB
from .style_gan import ConvBlock
import math, random

class SG2UGen(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        predict_offset=False
    ):
        super().__init__()
        self.predict_offset=predict_offset

        self.size = size

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style_emb = nn.Sequential(*layers)

        self.channels = {
            #4: 512,
            #8: 512,
            4: 512,
            8: 512,
            16: 256,# * channel_multiplier,
            32: 128 * channel_multiplier,
            64: 64 * channel_multiplier,
            128: 32 * channel_multiplier,
            256: 16 * channel_multiplier,
        }
        self.down_channels = {
            4: 256,
            8: 256,
            16: 128,
            32: 64,
            64: 32,
            128: 16, 
            256: 8 
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            down_out_channel = self.down_channels[2 ** (i-1)]

            self.convs.append(
                StyledConv(
                    in_channel+down_out_channel, #little addition here for U append
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

        #Added QR image layers
        self.in_conv = nn.Sequential(
                nn.Conv2d(1,self.down_channels[256],7,padding=3),
                nn.BatchNorm2d(self.down_channels[256]),
                nn.LeakyReLU(0.1),
                )
        fused=True
        self.down_convs = nn.ModuleList(
            [
                ConvBlock( self.down_channels[256], self.down_channels[128], 3, 1, downsample=True, fused=fused), #128
                ConvBlock(self.down_channels[128], self.down_channels[64], 3, 1, downsample=True, fused=fused), #64
                ConvBlock(self.down_channels[64], self.down_channels[32], 3, 1, downsample=True, fused=fused), #32
                ConvBlock(self.down_channels[32], self.down_channels[16], 3, 1, downsample=True), #16
                ConvBlock(self.down_channels[16], self.down_channels[8], 3, 1, downsample=True), #8
                ConvBlock(self.down_channels[8], self.down_channels[4], 3, 1, downsample=True), #4
            ])

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style_emb(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style_emb(input)

    def forward(
        self,
        qr_image,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        x = self.in_conv(qr_image)
        prev_xs=[]
        for down_conv in self.down_convs:
            prev_xs.append(x)
            #if self.all_style:
            #    x,_ = down_conv((x,style))
            #else:
            x = down_conv(x)
        prev_xs.append(x)
        prev_xs.reverse()

        if not input_is_latent:
            styles = [self.style_emb(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            #print('out {}'.format(out.size()))
            #print('prev {}'.format(prev_xs[i//2].size()))
            out = torch.cat((out,prev_xs[i//2]),dim=1)   
            
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = F.tanh(skip)
        if self.predict_offset:
            image += qr_image

        if return_latents:
            return image, latent

        else:
            return image
