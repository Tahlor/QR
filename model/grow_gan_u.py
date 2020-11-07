import torch.nn as nn
import torch.nn.functional as F

from .style_gan import StyledConvBlock, EqualConv2d, PixelNorm, EqualLinear

class GrowGen(nn.Module):
    def __init__(self, code_dim, fused=True, n_mlp=6):
        super().__init__()
        self.style_dim = code_dim

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
                StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
                StyledConvBlock(512, 256, 3, 1, upsample=True),  # 16
                StyledConvBlock(256, 128, 3, 1, upsample=True),  # 32
                StyledConvBlock(128, 64, 3, 1, upsample=True),  # 64
                StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 128
                StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 256
                #StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
                #StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                #EqualConv2d(512, 3, 1),
                #EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(512, 3, 1),
                EqualConv2d(256, 3, 1),
                EqualConv2d(128, 3, 1),
                EqualConv2d(64, 3, 1),
                EqualConv2d(32, 3, 1),
                EqualConv2d(16, 3, 1),
            ]
        )

        self.down_convs = nn.ModuleList(
            ConvBlock( 8, 16, 3, 1, downsample=True, fused=fused), #128
            ConvBlock(16, 32, 3, 1, downsample=True, fused=fused), #64
            ConvBlock(32, 64, 3, 1, downsample=True, fused=fused), #32
            ConvBlock(64, 128, 3, 1, downsample=True), #16
            ConvBlock(128, 256, 3, 1, downsample=True), #8
            ConvBlock(256, 256, 3, 1, downsample=True), #4
            ])

        # self.blur = Blur()
        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style_emb = nn.Sequential(*layers)

    def forward(self, qr_image, style=None, step=0, alpha=-1, mixing_range=(-1, -1)):
        x = self.in_conv(qr_image)
        prev_xs=[]
        for down_conv in self.down_convs:
            prev_xs.append(x)
            if self.all_style:
                x,_ = down_conv((x,style))
            else:
                x = down_conv(x)
        prev_xs.reverse()
        batch_size = qr_image.size(0)
        #if style=='mixing' and random.random()<0.9:
        #    styleA, styleB = torch.randn(2,batch_size, self.style_dim, device='cuda').chunk(2,dim=0)
        #    style = [styleA.squeeze(0),styleB.squeeze(0)]
        #elif style is None or type(style) is str:
        style = torch.randn(batch_size, self.style_dim, device='cuda')
        style = [style.squeeze(0)]

        #out = noise[0]
        out = torch.randn(batch_size, 1, 4, 4, device=qr_image.device)

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(random.sample(list(range(step)), len(style) - 0))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(style))

                style_step = style[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = style[1]

                else:
                    style_step = style[0]

            if i > 0 and step > 0:
                out_prev = out

            out = torch.cat((out,prev_xs),dim=1)   
            out = conv(out, style_step)#, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out
