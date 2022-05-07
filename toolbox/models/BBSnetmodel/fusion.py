import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
#1,2,3,4层

# class FilterLayer(nn.Module):
#     def __init__(self, in_planes, out_planes, reduction=16):
#         super(FilterLayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_planes, out_planes // reduction),
#             nn.ReLU(inplace=True),
#             nn.Linear(out_planes // reduction, out_planes),
#             nn.Sigmoid()
#         )
#         self.out_planes = out_planes
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, self.out_planes, 1, 1)
#         return y
# #ACM
# class acm(nn.Module):
#     def __init__(self,num_channel):
#
#         super(acm, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv2d(num_channel,num_channel,kernel_size=1)
#         self.activation = nn.Sigmoid()
#     def forward(self,x):
#         aux = self.pool(x)
#         aux = self.conv(aux)
#         aux = self.activation(aux)
#         return x*aux
# class yfusion(nn.Module):
#     def __init__(self,pre_inchannel,inchannel):
#         super(yfusion, self).__init__()
#
#         self.acm = acm(inchannel)
#
#
#         self.conv_cat = nn.Conv2d(2*inchannel,inchannel,kernel_size=1,padding=0)
#         self.bn = nn.BatchNorm2d(inchannel)
#         self.relu = nn.ReLU()
#         #self.conv_cat2 = nn.Conv2d(2*inchannel,inchannel,kernel_size=3,padding=1)
#         self.activation = nn.Sigmoid()
#         self.conv_cat_r = nn.Conv2d(inchannel,1,kernel_size=1,padding=0)
#         self.bn_r = nn.BatchNorm2d(1)
#         self.conv_cat_d = nn.Conv2d(inchannel,1,kernel_size=1,padding=0)
#         self.bn_d = nn.BatchNorm2d(1)
#         self.filter = FilterLayer(2*inchannel,inchannel)
#         # self.conv_to1 = nn.Conv2d(inchannel,1,kernel_size=1,padding=0)
#         self.softmax = nn.Softmax(dim=1)
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()
#         self.trans = nn.Conv2d(inchannel,inchannel,kernel_size=3,stride=1,padding=1)
#         self.bn2 = nn.BatchNorm2d(inchannel)
#         # self.bn_to1 = nn.BatchNorm2d(1)
#         self.relu3 = nn.ReLU()
#     def forward(self,rgb,depth,pre_rgb):
#         rgb_out = self.acm(rgb)
#         depth_out = depth
#         rd_out = torch.cat((rgb_out,depth_out),dim=1)
#         rd_out = self.conv_cat(rd_out)
#         rd_out = self.bn(rd_out)
#         rd_out = self.relu(rd_out)
#         pre_rgb = self.filter(pre_rgb)
#         weight = pre_rgb*rd_out
#
#         rgb_out = rgb_out+weight
#         depth_out = depth_out+weight
#
#         rgb_out = self.conv_cat_r(rgb_out)
#         rgb_out = self.bn_r(rgb_out)
#         depth_out = self.conv_cat_d(depth_out)
#         depth_out = self.bn_d(depth_out)
#         rgb_out = self.softmax(rgb_out)
#         depth_out = self.softmax(depth_out)
#
#         rgb_out = rgb*rgb_out
#         depth_out = depth*depth_out
#
#
#         F = rgb_out+depth_out
#         F = self.trans(F)
#         F = self.bn2(F)
#         F = self.relu3(F)
#         return F
#
#
# #特殊情况为第一层，没有pre_rgb输入
# class yfusion_layer4(nn.Module):
#     def __init__(self, inchannel):
#         super(yfusion_layer4, self).__init__()
#
#         self.acm = acm(inchannel)
#
#         self.conv_cat = nn.Conv2d(2 * inchannel, inchannel, kernel_size=1, padding=0)
#         self.bn = nn.BatchNorm2d(inchannel)
#         self.relu = nn.ReLU()
#         # self.conv_cat2 = nn.Conv2d(2*inchannel,inchannel,kernel_size=3,padding=1)
#         self.activation = nn.Sigmoid()
#         self.conv_cat_r = nn.Conv2d(inchannel, 1, kernel_size=1, padding=0)
#         self.bn_r = nn.BatchNorm2d(1)
#         self.conv_cat_d = nn.Conv2d(inchannel, 1, kernel_size=1, padding=0)
#         self.bn_d = nn.BatchNorm2d(1)
#         self.filter = FilterLayer(2 * inchannel, inchannel)
#         # self.conv_to1 = nn.Conv2d(inchannel,1,kernel_size=1,padding=0)
#         self.softmax = nn.Softmax(dim=1)
#         self.relu1 = nn.ReLU()
#         self.relu2 = nn.ReLU()
#         self.trans = nn.Conv2d(inchannel,inchannel,kernel_size=3,stride=1,padding=1)
#         self.bn2 = nn.BatchNorm2d(inchannel)
#         # self.bn_to1 = nn.BatchNorm2d(1)
#         self.relu3 = nn.ReLU()
#
#     def forward(self, rgb, depth):
#         rgb_out = self.acm(rgb)
#         depth_out = depth
#         rd_out = torch.cat((rgb_out, depth_out), dim=1)
#         rd_out = self.conv_cat(rd_out)
#         rd_out = self.bn(rd_out)
#         rd_out = self.activation(rd_out)
#         #pre_rgb = self.filter(pre_rgb)
#         #weight = pre_rgb * rd_out
#
#         rgb_out = rgb_out + rd_out
#         depth_out = depth_out + rd_out
#
#         rgb_out = self.conv_cat_r(rgb_out)
#         rgb_out = self.bn_r(rgb_out)
#         depth_out = self.conv_cat_d(depth_out)
#         depth_out = self.bn_d(depth_out)
#
#         rgb_out = self.softmax(rgb_out)
#         depth_out = self.softmax(depth_out)
#
#         rgb_out = rgb * rgb_out
#         depth_out = depth * depth_out
#
#         F = rgb_out + depth_out
#         F = self.trans(F)
#         F = self.bn2(F)
#         F = self.relu3(F)
#         return F

class FilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(FilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y
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
class AR(nn.Module):
    def __init__(self,inchannel):
        super(AR, self).__init__()
        # self.conv = BasicConv2d(in_channel = 2*inchannel,out_channel = inchannel,kernel_size=3,padding=1)
        self.conv13 = BasicConv2d(in_channel=inchannel,out_channel=inchannel,kernel_size=(1,3),padding=(0,1))
        self.conv31 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1), padding=(1, 0))

        self.conv13_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(1, 3), padding=(0, 1))
        self.conv31_2 = BasicConv2d(in_channel=inchannel, out_channel=inchannel, kernel_size=(3, 1),padding=(1, 0))
        # self.aux_conv = nn.Conv2d(inchannel,inchannel,kernel_size=3,padding=1)
        self.aux_conv = FilterLayer(inchannel,inchannel)
        self.bn1 = nn.BatchNorm2d(inchannel)
        self.sof = nn.Softmax(dim=1)
        self.fuseconv = BasicConv2d(inchannel*2,inchannel,kernel_size=3,padding=1)
        self.conv_end = nn.Conv2d(2*inchannel,inchannel,kernel_size=3,padding=1)
        # self.bn2 = nn.BatchNorm2d(inchannel)
    def forward(self,max,aux):
        max_1 = self.conv13(max)
        max_1 = self.conv31(max_1)

        max_2 = self.conv31_2(max)
        max_2 = self.conv13_2(max_2)
        fuse_max = torch.cat((max_1, max_2), dim=1)
        fuse_max = self.fuseconv(fuse_max)
        aux_w = self.aux_conv(aux)

        weight = aux_w*fuse_max
        max_1 = weight+max_1
        max_2 = weight+max_2
        ar_out = torch.cat((max_1,max_2),dim=1)
        ar_out = self.conv_end(ar_out)
        ar_out = self.bn1(ar_out)
        ar_out = self.sof(ar_out)
        ar_out = ar_out*max
        return ar_out
# class acm(nn.Module):
#     def __init__(self,num_channel):
#
#         super(acm, self).__init__()
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv2d(num_channel,num_channel,kernel_size=1)
#         self.activation = nn.Sigmoid()
#     def forward(self,x):
#         aux = self.pool(x)
#         aux = self.conv(aux)
#         aux = self.activation(aux)
#         return x*aux

class ER(nn.Module):
    def __init__(self, in_channel):
        super(ER, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))

        self.b_1 = BasicConv2d(in_channel * 3, in_channel, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channel,in_channel,kernel_size=1,padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):

        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))
        out = self.relu(buffer_1+self.conv_res(x))

        return out
class fusion(nn.Module):
    def __init__(self,inc):
        super(fusion, self).__init__()
        self.ar = AR(inchannel=inc)
        # self.a = acm(num_channel=inc)
        # self.conv_end = BasicConv2d(in_channel=inc*2,out_channel=inc,kernel_size=3,padding=1)
        self.sof = nn.Softmax(dim=1)
        self.er = ER(in_channel=inc)
    def forward(self,r,d):

        br = self.ar(r,d)
        bd = self.ar(d,r)
        br = self.sof(br)
        bd = self.sof(bd)
        br = br*r
        bd = bd*d
        out = br+bd

        out = self.er(out)
        return out

