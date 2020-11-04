
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pure_gen import StyledConvBlock, PixelNorm, EqualConv2d, ArgTwoId

class ConvGen(nn.Module):
    def __init__(self, style_size, dim=16, down_steps=4, n_style_trans=6,all_style=False):
        super(ConvGen, self).__init__()
        fused=True
        self.all_style=all_style

        self.down_convs = nn.ModuleList()
        for i in range(down_steps):
            dim*=2
            if all_style:
                self.down_convs.append( nn.Sequential(
                        ArgTwoId(nn.MaxPool2d(2)),
                        StyledConvBlock(dim//2,dim,fused=fused,style_dim=style_size)
                        ))
            else:
                self.down_convs.append( nn.Sequential(
                        nn.MaxPool2d(2), 
                        nn.Conv2d(dim//2,dim,3,padding=1),
                        nn.BatchNorm2d(dim),
                        nn.LeakyReLU(0.1),
                        nn.Conv2d(dim,dim,3,padding=1),
                        nn.BatchNorm2d(dim),
                        nn.LeakyReLU(0.1)
                        ))
        self.up_convs = nn.ModuleList()
        for i in range(down_steps):
            self.up_convs.append( nn.Sequential(
                    StyledConvBlock(2*dim if i!=0 else dim,dim,upsample=False,style_dim=style_size),
                    StyledConvBlock(dim,dim//2,upsample=True,fused=fused,style_dim=style_size)
                    ))
            dim=dim//2
        self.up_convs.append(StyledConvBlock(2*dim,dim,upsample=False,style_dim=style_size))

        self.in_conv = nn.Sequential(
                nn.Conv2d(1,dim,7,padding=3),
                nn.BatchNorm2d(dim),
                nn.LeakyReLU(0.1),
                #nn.Conv2d(dim,dim,3,padding=1),
                #nn.BatchNorm2d(dim),
                #nn.LeakyReLU(0.1)
                )
        self.out_conv = nn.Sequential(
                EqualConv2d(dim, 3, 1),
                nn.Tanh()
                )


        layers = [PixelNorm()]
        for i in range(n_style_trans):
            layers.append(nn.Linear(style_size, style_size))
            layers.append(nn.LeakyReLU(0.2))

        self.style_emb = nn.Sequential(*layers)

    def forward(self, qr_img,style):
        style = self.style_emb(style)
        x = self.in_conv(qr_img)
        prev_xs=[]
        for down_conv in self.down_convs:
            prev_xs.append(x)
            if self.all_style:
                x,_ = down_conv((x,style))
            else:
                x = down_conv(x)
        y,_ = self.up_convs[0]((x,style))
        #for i in range(len(self.up_convs)-2,-1,-1):
        #    y,_ = self.up_convs[i]((torch.cat([prev_xs[i],y],dim=1),style))
        prev_xs.reverse()
        for up_conv,x in zip(self.up_convs[1:],prev_xs):
            y,_ = up_conv((torch.cat([x,y],dim=1),style))
        y = self.out_conv(y)

        return y+qr_img

