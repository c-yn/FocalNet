import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)


    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x
    
class ResBlock1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock1, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main1 = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2):
        x1 = self.main(x1) + x1 
        x2 = self.main1(x2) + x2
        return x1, x2

class UNet(nn.Module):
    def __init__(self, inchannel, outchannel, num_res) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(num_res):
            self.layers.append(ResBlock1(inchannel//2, outchannel//2))
        self.num_res = num_res
        self.down = nn.Conv2d(inchannel//2, outchannel//2, kernel_size=2, stride=2, groups=inchannel//2)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x1, x2 = torch.chunk(x, 2, dim=1)
                x2 = self.down(x2)
                x1, x2 = layer(x1, x2)

            elif i == self.num_res - 1:
                x1, x2 = layer(x1, x2)
                x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear')
                x = torch.cat((x1,x2), dim=1)
                
            else:
                x1, x2 = layer(x1, x2)
                
        return x
