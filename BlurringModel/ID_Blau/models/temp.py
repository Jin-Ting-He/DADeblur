import torch
import torch.nn as nn
import logging
import sys
from thop import profile
from models import adaptive_instance_normalization as AdaIn

class Encoder(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4, activation = nn.LeakyReLU(0.2, True)):
        super(Encoder, self).__init__()

        self.activation = activation

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        
        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_layer3_2 = nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1))
        self.en_layer3_3 = nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1))
        self.en_layer3_4 = nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1))

        self.en_layer4_1 = nn.Sequential(
            nn.Conv2d(dim_3, dim_4, kernel_size=3, stride=2, padding=1),
            self.activation)


    def forward(self, x):

        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        hx = self.activation(self.en_layer1_4(hx) + hx)
        residual_1 = hx  # b, c, h, w

        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        hx = self.activation(self.en_layer2_4(hx) + hx)
        residual_2 = hx  # b, c, h/2, w/2

        hx = self.en_layer3_1(hx)
        hx = self.activation(self.en_layer3_2(hx) + hx)
        hx = self.activation(self.en_layer3_3(hx) + hx)
        hx = self.activation(self.en_layer3_4(hx) + hx)
        residual_3 = hx  # b, c, h/2, w/2

        hx = self.en_layer4_1(hx)

        return hx, residual_1, residual_2, residual_3


class Decoder(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, dim_4, activation= nn.LeakyReLU(0.2, True)):
        super(Decoder, self).__init__()
        self.activation = activation

        self.de_layer4_1 = nn.Sequential(
            nn.ConvTranspose2d(dim_4, dim_3, kernel_size=4, stride=2, padding=1),
            self.activation)
        
        self.de_layer3_1 = nn.Sequential(
            nn.Conv2d(dim_3+dim_3, dim_3, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1))
        self.de_layer3_2 = nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1))
        self.de_layer3_3 = nn.Sequential(
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_3, dim_3, kernel_size=3, padding=1))
        self.de_layer3_4 = nn.Sequential(
            nn.ConvTranspose2d(dim_3, dim_2, kernel_size=4, stride=2, padding=1),
            self.activation)

        self.de_layer2_1 = nn.Sequential(
            nn.Conv2d(dim_2+dim_2, dim_2, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.de_layer2_3 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.de_layer2_4 = nn.Sequential(
            nn.ConvTranspose2d(dim_2, dim_1, kernel_size=4, stride=2, padding=1),
            self.activation)

        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(dim_1+dim_1, dim_1, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_4 = nn.Sequential(
            nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
        )

    def forward(self, x, residual_1, residual_2, residual_3):
        hx = self.de_layer4_1(x)

        hx = self.activation(self.de_layer3_1(torch.cat((hx, residual_3), dim=1)) + hx)
        hx = self.activation(self.de_layer3_2(hx)+hx)
        hx = self.activation(self.de_layer3_3(hx)+hx)
        hx = self.de_layer3_4(hx)

        hx = self.activation(self.de_layer2_1(torch.cat((hx, residual_2), dim=1)) + hx)
        hx = self.activation(self.de_layer2_2(hx)+hx)
        hx = self.activation(self.de_layer2_3(hx)+hx)
        hx = self.de_layer2_4(hx)

        hx = self.activation(self.de_layer1_1(torch.cat((hx, residual_1), dim=1)) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.activation(self.de_layer1_3(hx) + hx)
        hx = self.de_layer1_4(hx)

        return hx

class BlurEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=6, activation=nn.LeakyReLU(0.2, True)):
        super().__init__()
        self.out_channels = out_channels * 2
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(block_num):
            self.layers.append(nn.Linear(in_channels if i == 0 else self.out_channels, self.out_channels))
    
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = self.activation(layer(output))

        return output.reshape(-1, int(self.out_channels/2), 2)
    
class AdainResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(0.2, True)):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            activation,
        )
        self.layer2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, x, style):
        output = self.layer1(x)
        output = AdaIn(output, style)
        output = self.layer2(output) + x

        return output

class Reblur_Model(nn.Module):
    def __init__(self, dim_1=32, dim_2=64, dim_3=128, dim_4=256, adain_residual_block_num=9, activation=nn.LeakyReLU(0.2, True)):
        super(Reblur_Model, self).__init__()
        self.encoder = Encoder(dim_1, dim_2, dim_3, dim_4, activation=activation)
        self.decoder = Decoder(dim_1, dim_2, dim_3, dim_4, activation=activation)
        self.middle = nn.ModuleList()
        for _ in range(adain_residual_block_num):
            self.middle.append(
                AdainResidualBlock(dim_4, dim_4, activation=activation)
            )
        self.blurEmbedding = BlurEmbedding(1, dim_4, block_num=6, activation=activation)

    def forward(self, x, ratio):
        # x: b, c, h, w
        ratio_embedding = self.blurEmbedding(ratio)
        hx, residual_1, residual_2, residual_3 = self.encoder(x)
        for layer in self.middle:
            hx = layer(hx, ratio_embedding)
        hx = self.decoder(hx, residual_1, residual_2, residual_3)

        return hx + x

if __name__ == '__main__':
    # Debug
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = Reblur_Model().cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    ratio = torch.tensor([0.]).cuda().reshape(-1,1)
    print("ratio",ratio)
    flops, params = profile(net, (input, ratio))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

import torch
import torch.nn as nn
import logging
import sys
from thop import profile
from models import adaptive_instance_normalization as AdaIn

class Encoder(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, activation):
        super(Encoder, self).__init__()

        self.activation = activation

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            self.activation)
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            self.activation)
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))

        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=2, padding=1),
            self.activation)


    def forward(self, x):

        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        hx = self.activation(self.en_layer1_4(hx) + hx)
        residual_1 = hx  # b, c, h, w

        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        hx = self.activation(self.en_layer2_4(hx) + hx)
        residual_2 = hx  # b, c, h/2, w/2

        hx = self.en_layer3_1(hx)

        return hx, residual_1, residual_2


class Decoder(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3, activation):
        super(Decoder, self).__init__()

        self.activation = activation

        self.de_layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(dim_3, dim_2, kernel_size=4, stride=2, padding=1),
            self.activation)

        self.de_layer2_1 = nn.Sequential(
            nn.Conv2d(dim_2+dim_2, dim_2, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.de_layer2_3 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.de_layer2_4 = nn.Sequential(
            nn.ConvTranspose2d(dim_2, dim_1, kernel_size=4, stride=2, padding=1),
            self.activation)

        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(dim_1+dim_1, dim_1, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_4 = nn.Sequential(
            nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
        )

    def forward(self, x, residual_1, residual_2):
        hx = self.de_layer3_1(x)

        hx = self.activation(self.de_layer2_1(torch.cat((hx, residual_2), dim=1)) + hx)
        hx = self.activation(self.de_layer2_2(hx)+hx)
        hx = self.activation(self.de_layer2_3(hx)+hx)
        hx = self.de_layer2_4(hx)

        hx = self.activation(self.de_layer1_1(torch.cat((hx, residual_1), dim=1)) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.activation(self.de_layer1_3(hx) + hx)
        hx = self.de_layer1_4(hx)

        return hx

class BlurEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=6, activation=nn.LeakyReLU(0.2, True)):
        super().__init__()
        self.out_channels = out_channels
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(block_num):
            self.layers.append(nn.Linear(in_channels if i == 0 else self.out_channels, self.out_channels))
    
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = self.activation(layer(output))

        return output
    
class AdainResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(0.2, True)):
        super().__init__()
        self.out_channels = out_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            activation,
        )
        self.layer2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
    
    def forward(self, x, style):
        style = style.reshape(-1, self.out_channels, 2)
        output = self.layer1(x)
        output = AdaIn(output, style)
        output = self.layer2(output) + x

        return output

class Reblur_Model(nn.Module):
    def __init__(self, dim_1=32, dim_2=64, dim_3=128, adain_residual_block_num=9, activation=nn.LeakyReLU(0.2, True)):
        super(Reblur_Model, self).__init__()
        self.encoder = Encoder(dim_1, dim_2, dim_3, activation=activation)
        self.decoder = Decoder(dim_1, dim_2, dim_3, activation=activation)
        self.middle = nn.ModuleList()
        for _ in range(adain_residual_block_num):
            self.middle.append(
                AdainResidualBlock(dim_3, dim_3, activation=activation)
            )
        self.blurEmbedding = BlurEmbedding(1, dim_3, block_num=6, activation=activation)
        self.blurEmbedding_1 = nn.Sequential(
            nn.Linear(dim_2+dim_2, dim_1+dim_1),
            activation)
        self.blurEmbedding_2 = nn.Sequential(
            nn.Linear(dim_3+dim_3, dim_2+dim_2),
            activation)
        self.AdainResidualBlock_1 = AdainResidualBlock(dim_1, dim_1, activation=activation)
        self.AdainResidualBlock_2 = AdainResidualBlock(dim_2, dim_2, activation=activation)

    def forward(self, x, ratio):
        # x: b, c, h, w
        ratio_embedding = self.blurEmbedding(ratio)
        ratio_embedding_2 = self.blurEmbedding_2(ratio_embedding)
        ratio_embedding_1 = self.blurEmbedding_1(ratio_embedding_2)

        hx, residual_1, residual_2 = self.encoder(x)
        residual_1 = self.AdainResidualBlock_1(residual_1, ratio_embedding_1)
        residual_2 = self.AdainResidualBlock_2(residual_2, ratio_embedding_2)

        for layer in self.middle:
            hx = layer(hx, ratio_embedding)
        hx = self.decoder(hx, residual_1, residual_2)

        return hx + x


if __name__ == '__main__':
    # Debug
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = Reblur_Model().cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    ratio = torch.tensor([0.]).cuda().reshape(-1,1)
    print("ratio",ratio)
    flops, params = profile(net, (input, ratio))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        b, c, h, w = x.shape
        assert h % 2 == 0 or w % 2 == 0, "w and h must be even"
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x):
        return self.upsample(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, activatedfun=nn.SiLU):
        super().__init__()
        self.layer1 = nn.Sequential(
            activatedfun(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        self.layer2 = nn.Sequential(
            activatedfun(),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output) + self.shortcut(x)
        
        return output
        


class UNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, channel_mults=(1, 2, 4), num_res_blocks=2, time_dim=128 * 4, activatedfun=nn.SiLU, dropout=0.1, num_groups=32):
        super().__init__()
    
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.downblocks = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResidualBlock(now_channels, out_channels, dropout, activatedfun=activatedfun)
                )
                now_channels = out_channels
                channels.append(now_channels)
            
            if i != len(channel_mults) - 1:
                self.downblocks.append(Downsample(now_channels))
                channels.append(now_channels)
        

        self.mid = nn.ModuleList([
            ResidualBlock(now_channels, now_channels, dropout, activatedfun=activatedfun),
            ResidualBlock(now_channels, now_channels, dropout, activatedfun=activatedfun),
        ])

        self.upblocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult

            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResidualBlock(channels.pop() + now_channels, out_channels, dropout, activatedfun=activatedfun)
                )
                now_channels = out_channels
            
            if i != 0:
                self.upblocks.append(Upsample(now_channels))
        
        assert len(channels) == 0

        self.last_layer = nn.Sequential(
            nn.GroupNorm(num_groups, base_channels),
            activatedfun(),
            nn.Conv2d(base_channels, img_channels, 3, padding=1)
        )
    
    def forward(self, x, time):     
        time_emb = self.time_embedding(time)  
        x = self.init_conv(x)

        skips = [x]

        for layer in self.downblocks:
            x = layer(x, time_emb)
            skips.append(x)

        for layer in self.mid:
            x = layer(x, time_emb)
        
        for layer in self.upblocks:
            if isinstance(layer, ResidualBlock):
                x = torch.cat([x, skips.pop()], dim=1)
            x = layer(x, time_emb)

        x = self.last_layer(x)
        assert len(skips) == 0
        return x

if __name__ == '__main__':
    # Debug
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = Reblur_Net(dim_1=32, dim_2=64, dim_3=128).cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    ratio = torch.tensor([0.]).cuda().reshape(-1,1)
    print("ratio",ratio)
    flops, params = profile(net, (input, ratio))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')