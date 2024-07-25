# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .origin.from_origin import Backbone_ResNet50_in3, Backbone_VGG16_in3
from thop import profile

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
    
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.activation = nn.LeakyReLU(0.2, True)
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            self.activation,
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            self.activation
        )

    def forward(self, x):
        return self.main(x) + x

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        
        self.main = nn.Sequential(
            # nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            ResBlock(out_channel, out_channel),
            ResBlock(out_channel, out_channel),
            ResBlock(out_channel, out_channel),
            ResBlock(out_channel, out_channel),
            ResBlock(out_channel, out_channel),
        )
        
    def forward(self, x):
        return self.main(x)

class CM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CM, self).__init__()

        self.Up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=4, stride=2, padding=1),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel*2, out_channel, kernel_size=1),
            nn.LeakyReLU(0.2, True))

    def forward(self, x, y):
        x = self.Up(x)
        diffY = y.size()[2] - x.size()[2]
        diffX = y.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        z = torch.cat((x,y),dim=1)
        z = self.conv(z)
        return z

class AFF(nn.Module):
    def __init__(self):
        super(AFF, self).__init__()

        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(2048, 64, kernel_size=3, padding=1)

        self.conv0_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv1_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv2_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv3_1x1 = nn.Conv2d(192, 192, kernel_size=1)
        self.conv4_1x1 = nn.Conv2d(192, 192, kernel_size=1)

        self.conv0_out = nn.Conv2d(192, 32, kernel_size=3, padding=1)
        self.conv1_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv2_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv3_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)
        self.conv4_out = nn.Conv2d(192, 64, kernel_size=3, padding=1)

    def forward(self, x0,x1,x2,x3,x4):
        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # x0 fusion
        x1_up = F.interpolate(x1, scale_factor=2)
        x2_up = F.interpolate(x2, scale_factor=4)
        x0_out = self.conv0_out(self.conv0_1x1(torch.cat((x0, x1_up, x2_up),dim=1)))

        # x1 fusion
        x0_down = F.interpolate(x0, scale_factor=0.5)
        x2_up = F.interpolate(x2, scale_factor=2)
        x1_out = self.conv1_out(self.conv1_1x1(torch.cat((x0_down, x1, x2_up),dim=1)))

        # x2 fusion
        x1_down = F.interpolate(x1, scale_factor=0.5)
        x3_up = F.interpolate(x3, scale_factor=2)
        x2_out = self.conv2_out(self.conv2_1x1(torch.cat((x1_down, x2, x3_up),dim=1)))

        # x3 fusion
        x2_down = F.interpolate(x2, scale_factor=0.5)
        x4_up = F.interpolate(x4, scale_factor=2)

        diffY = x4_up.size()[2] - x2_down.size()[2]
        diffX = x4_up.size()[3] - x2_down.size()[3]

        x2_down = F.pad(x2_down, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x3_out = self.conv3_out(self.conv3_1x1(torch.cat((x2_down, x3, x4_up),dim=1)))

        # x4 fusion
        x2_down = F.interpolate(x2, scale_factor=0.25)
        x3_down = F.interpolate(x3, scale_factor=0.5)
        diffY = x4.size()[2] - x2_down.size()[2]
        diffX = x4.size()[3] - x2_down.size()[3]
        x2_down = F.pad(x2_down, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x4_out = self.conv4_out(self.conv4_1x1(torch.cat((x2_down, x3_down, x4),dim=1)))
        return x0_out, x1_out, x2_out, x3_out, x4_out
    
class MyNet_Res50_multiscale(nn.Module):
    def __init__(self):
        super(MyNet_Res50_multiscale, self).__init__()

        self.div_2, self.div_4, self.div_8, self.div_16, self.div_32 = Backbone_ResNet50_in3()
        self.aff = AFF()

        self.decoder_0 = Decoder(32, 32)
        self.decoder_1 = Decoder(64, 32)
        self.decoder_2 = Decoder(64, 64)
        self.decoder_3 = Decoder(64, 64)
        self.decoder_4 = Decoder(2048, 64)
        # self.decoder_out = Decoder(32,32)

        self.cm_0 = CM(32, 32)
        self.cm_1 = CM(64, 32)
        self.cm_2 = CM(64, 64)
        self.cm_3 = CM(64, 64)

        self.Up = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data):
        in_data_2 = self.div_2(in_data)
        in_data_4 = self.div_4(in_data_2)
        in_data_8 = self.div_8(in_data_4)
        in_data_16 = self.div_16(in_data_8)
        in_data_32 = self.div_32(in_data_16)
        # print(in_data_2.size(), in_data_4.size(), in_data_8.size(), in_data_16.size(), in_data_32.size())

        aff_x0, aff_x1, aff_x2, aff_x3, aff_x4 = self.aff(in_data_2, in_data_4, in_data_8, in_data_16, in_data_32)
        
        decode_data_32 = self.decoder_4(aff_x4)
        decode_data_16 = self.decoder_3(self.cm_3(decode_data_32, aff_x3))     
        decode_data_8 = self.decoder_2(self.cm_2(decode_data_16, aff_x2))  
        decode_data_4 = self.decoder_1(self.cm_1(decode_data_8, aff_x1))
        decode_data_2 = self.decoder_0(self.cm_0(decode_data_4, aff_x0))

        out = self.Up(decode_data_2)
        # out = self.decoder_out(out)
        out = self.classifier(out)
        
        return out
    
class MyNet_VGG16(nn.Module):
    def __init__(self):
        super(MyNet_VGG16, self).__init__()

        (
            self.encoder1,
            self.encoder2,
            self.encoder4,
            self.encoder8,
            self.encoder16,
        ) = Backbone_VGG16_in3()

        self.decoder_0 = Decoder(64, 32)
        self.decoder_1 = Decoder(128, 64)
        self.decoder_2 = Decoder(256, 128)
        self.decoder_3 = Decoder(512, 256)
        self.decoder_4 = Decoder(512, 512)

        self.aff = AFF()

        self.cm_0 = CM(64, 32)
        self.cm_1 = CM(128,128)
        self.cm_2 = CM(256,256)
        self.cm_3 = CM(512, 512)

        self.classifier = nn.Conv2d(32, 1, 1)

    def forward(self, in_data):
        in_data_1 = self.encoder1(in_data)
        in_data_2 = self.encoder2(in_data_1)
        in_data_4 = self.encoder4(in_data_2)
        in_data_8 = self.encoder8(in_data_4)
        in_data_16 = self.encoder16(in_data_8)

        decode_data_16 = self.decoder_4(in_data_16)
        decode_data_8 = self.decoder_3(self.cm_3(decode_data_16, in_data_8))     
        decode_data_4 = self.decoder_2(self.cm_2(decode_data_8, in_data_4))  
        decode_data_2 = self.decoder_1(self.cm_1(decode_data_4, in_data_2))
        decode_data_1 = self.decoder_0(self.cm_0(decode_data_2, in_data_1))

        out = self.classifier(decode_data_1)
        
        return out

if __name__ == '__main__':
    # Debug
    print("Test")
    net = MyNet_Res50_multiscale().cuda()
    input = torch.randn((1, 3, 320, 320)).cuda()

    """
    for i in range(100):
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            output, _ = net(input, input_quarter)
            torch.cuda.synchronize()
            stop = time.time()
            print((stop-start) / output.shape[1], output.shape[1])
    """
    flops, params = profile(net, (input, ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')