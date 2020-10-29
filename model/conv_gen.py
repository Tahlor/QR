
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pure_gen import StyledConvBlock, PixelNorm

class ConvGen(nn.Module):
    def __init__(self, style_size, dim=16, down_steps=4, n_style_trans=6)
        super(ConvGen, self).__init__()
        fused=True

        self.down_convs = nn.ModuleList()
        for i in range(down_steps):
            dim*=2
            self.down_convs.append( nn.Sequentail(
                    nn.MaxPool(2), 
                    nn.Conv2d(dim//2,dim,3,padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(dim,dim,3,padding=1)
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU(0.1)
                    ))
        self.up_convs = nn.ModuleList()
        for i in range(down_steps):
            self.up_convs.append( nn.Sequentail(
                    StyledConvBlock(dim,dim,upsample=False,style_dim=style_size)
                    StyledConvBlock(dim,dim//2,upsample=True,fused=fused,style_dim=style_size)
                    ))
            dim=dim//2

        self.in_convs = nn.Sequentail(
                nn.Conv2d(1,dim,7,padding=3),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(0.1),
                #nn.Conv2d(dim,dim,3,padding=1),
                #nn.BatchNorm2d(dim),
                #nn.LeakyReLU(0.1)
                )
        self.out_convs = nn.Sequentail(
                StyledConvBlock(dim,dim,upsample=False,style_dim=style_size),
                EqualConv2d(dim, 3, 1),
                nn.Tanh()
                )


        layers = [PixelNorm()]
        for i in range(n_style_trans):
            layers.append(nn.Linear(style_size, style_size))
            layers.append(nn.LeakyReLU(0.2))

        self.style_emb = nn.Sequential(*layers)

    def forward(self, qr_img,style,mask=None,return_intermediate=False): #, noise=None):
        style = self.style_emb(style)
        x,_ = self.in_conv((qr_img,style))
        prev_xs=[]
        for down_conv in self.down_convs:
            prev_xs.append(x)
            x,_ = self.conv2(x)
        y = self.up_convs[-1]((x,style))
        for i in range(len(self.up_convs)-2,-1,-1):
            y = self.up_convs[i]((torch.cat([prev_xs[i],y],dim=1),style))
        y = self.out_conv((torch.cat([prev_xs[0],y],dim=1),style))

        return y+qr_img

