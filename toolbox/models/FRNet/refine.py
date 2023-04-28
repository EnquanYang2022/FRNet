import torch
import torch.nn as nn
class BasicConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0,dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class Refine(nn.Module):
    def __init__(self,cur_channel,hig_channel,k):
        super(Refine, self).__init__()
        self.conv_t = BasicConv2d(hig_channel,cur_channel,kernel_size=3,padding=1)
        self.upsample = nn.Upsample(scale_factor=k, mode='bilinear', align_corners=True)
        self.corr_conv = nn.Conv2d(cur_channel,cur_channel,kernel_size=3,padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sig = nn.Sigmoid()

    def forward(self,current,higher):
        higher = self.upsample(higher)
        higher = self.conv_t(higher)
        corr = higher-current
        corr = self.corr_conv(corr)
        corr = self.avgpool(corr)
        corr = self.sig(corr)
        corr = higher*corr
        current = current+corr
        return current