import torch
import torch.nn as nn
from thop import profile
from .function import adaptive_instance_normalization as AdaIn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dims, dim_1, dim_2, dim_3, activation):
        super(Encoder, self).__init__()

        self.activation = activation

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(input_dims, dim_1, kernel_size=3, padding=1),
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

class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x4):
        x = torch.cat([x1, x2, x4], dim=1)
        return self.conv(x)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class BlurEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, block_num=9, activation=nn.LeakyReLU(0.2, True)):
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
        self.out_channels = out_channels
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            activation,
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            activation,
        )
    
    def forward(self, x, style):
        output = self.layer1(x)
        separate_index = int(self.out_channels / 2)
        output_f, output_b = output[:,:separate_index], output[:, separate_index:]
        output_f = AdaIn(output_f, style)
        output = torch.cat([output_f, output_b], dim=1)
        output = self.layer2(output) + x

        return output

class Reblur_Model(nn.Module):
    def __init__(self, input_dims=3, dim_1=32, dim_2=64, dim_3=128, adain_residual_block_num=9, activation=nn.LeakyReLU(0.2, True)):
        super(Reblur_Model, self).__init__()
        self.encoder = Encoder(input_dims, dim_1, dim_2, dim_3, activation=activation)
        self.decoder = Decoder(dim_1, dim_2, dim_3, activation=activation)
        self.middle = nn.ModuleList()
        for _ in range(adain_residual_block_num):
            self.middle.append(
                AdainResidualBlock(dim_3, dim_3, activation=activation)
            )
        self.blurEmbedding = BlurEmbedding(1, int(dim_3 / 2), block_num=6, activation=activation)
        self.AFFs = nn.ModuleList([
            AFF(dim_1 * 7, dim_1*1),
            AFF(dim_1 * 7, dim_1*2)
        ])

    def forward(self, x, ratio):
        # x: b, c, h, w
        ratio_embedding = self.blurEmbedding(ratio)
        hx, residual_1, residual_2 = self.encoder(x)
        for layer in self.middle:
            hx = layer(hx, ratio_embedding)

        z12 = F.interpolate(residual_1, scale_factor=0.5)
        z21 = F.interpolate(residual_2, scale_factor=2)
        z42 = F.interpolate(hx, scale_factor=2)
        z41 = F.interpolate(z42, scale_factor=2)

        residual_2 = self.AFFs[1](z12, residual_2, z42)
        residual_1 = self.AFFs[0](residual_1, z21, z41)

        hx = self.decoder(hx, residual_1, residual_2)

        return hx + x[:3]
    
if __name__ == '__main__':
    # Debug
    #logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    net = Reblur_Model(dim_1=32, dim_2=64, dim_3=128).cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    ratio = torch.tensor([0.]).cuda().reshape(-1,1)
    print("ratio",ratio)
    flops, params = profile(net, (input, ratio))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')