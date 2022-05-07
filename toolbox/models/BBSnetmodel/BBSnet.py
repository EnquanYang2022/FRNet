import torch
import torch as t
import torch.nn as nn
from toolbox.models.BBSnetmodel.decoder import SG
from torch.autograd import Variable as V
import torchvision.models as models
from toolbox.models.BBSnetmodel.ResNet import ResNet50,ResNet34
from torch.nn import functional as F
from toolbox.models.BBSnetmodel.fusion import fusion
from toolbox.models.BBSnetmodel.refine import Refine
from toolbox.models.BBSnetmodel.SG import SG
from toolbox.models.BBSnetmodel.ASPP import ASPP
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
class BasicConv2d_norelu(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0,dilation=1):
        super(BasicConv2d_norelu, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        # self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        # x = self.relu(x)
        return x

#GCM
# class GCM(nn.Module):
#     def __init__(self,inchannels,outchannels):
#         super(GCM, self).__init__()
#         self.branches0 = nn.Sequential(
#             BasicConv2d(inchannels,outchannels,kernel_size=1)
#         )
#         self.branches1 = nn.Sequential(
#             BasicConv2d(inchannels,outchannels,kernel_size=1),
#             BasicConv2d(outchannels,outchannels,kernel_size=(1,3),padding=(0,1)),
#             BasicConv2d(outchannels,outchannels,kernel_size=(3,1),padding=(1,0)),
#             BasicConv2d(outchannels,outchannels,kernel_size=3,padding=3,dilation=3)
#         )
#         self.branches2 = nn.Sequential(
#             BasicConv2d(inchannels, outchannels, kernel_size=1),
#             BasicConv2d(outchannels, outchannels, kernel_size=(1, 5), padding=(0, 2)),
#             BasicConv2d(outchannels, outchannels, kernel_size=(5, 1), padding=(2, 0)),
#             BasicConv2d(outchannels, outchannels, kernel_size=3, padding=5, dilation=5)
#         )
#         self.branches3 = nn.Sequential(
#             BasicConv2d(inchannels, outchannels, kernel_size=1),
#             BasicConv2d(outchannels, outchannels, kernel_size=(1, 7), padding=(0, 3)),
#             BasicConv2d(outchannels, outchannels, kernel_size=(7, 1), padding=(3, 0)),
#             BasicConv2d(outchannels, outchannels, kernel_size=3, padding=7, dilation=7)
#         )
#         self.conv1 = BasicConv2d(4*outchannels,outchannels,kernel_size=3,padding=1)
#         self.conv2 = BasicConv2d(inchannels,outchannels,kernel_size=1)
#     def forward(self,x):
#         x0 = self.branches0(x)
#         x1 = self.branches1(x)
#         x2 = self.branches2(x)
#         x3 = self.branches3(x)
#         out_cat = self.conv1(torch.cat((x0,x1,x2,x3),dim=1))
#         out_x = self.conv2(x)
#         out = out_cat+out_x
#         return out



#用rgb增强depth
# class DA(nn.Module):
#     def __init__(self,inchannel,outchannel):
#         super(DA, self).__init__()
#         self.conv1 = BasicConv2d(in_channel=2*inchannel,out_channel=outchannel,kernel_size=3,padding=1)
#         self.conv2 = nn.Conv2d(outchannel,outchannel,kernel_size=1,padding=0)
#         self.bn1 = nn.BatchNorm2d(outchannel)
#     def forward(self,r,d):
#         combine = torch.cat((r,d),dim=1)
#         combine = self.conv1(combine)
#         out = combine+r
#         out = self.conv2(out)
#         out = self.bn1(out)
#         out = out+d
#         return out

class serialaspp(nn.Module):
    def __init__(self,inc,outc,flag = None):
        super(serialaspp, self).__init__()
        # self.dconv1 = BasicConv2d_norelu(in_channel=2048,out_channel=1024,kernel_size=3,padding=1)
        # self.dconv6 = BasicConv2d_norelu(in_channel=1024,out_channel=512,kernel_size=3,padding=6,dilation=6)
        # self.dconv12 = BasicConv2d_norelu(in_channel=512,out_channel=256,kernel_size=3,padding=12,dilation=12)
        # self.dconv18 = BasicConv2d_norelu(in_channel=256,out_channel=64,kernel_size=3,padding=18,dilation=18)
        # self.dconv24 = BasicConv2d_norelu(in_channel=128,out_channel=64,kernel_size=3,padding=24,dilation=24)
        self.flag = flag
        self.dconv1 = BasicConv2d(in_channel=256, out_channel=256, kernel_size=3, padding=1)
        self.dconv2 = BasicConv2d(in_channel=128, out_channel=128, kernel_size=3, padding=2,dilation=2)
        self.dconv4 = BasicConv2d(in_channel=64, out_channel=64, kernel_size=3, padding=4,dilation=4)
        # self.dconv6 = BasicConv2d_norelu(in_channel=256, out_channel=128, kernel_size=3, padding=6, dilation=6)
        # self.dconv12 = BasicConv2d_norelu(in_channel=128, out_channel=64, kernel_size=3, padding=12, dilation=12)
        # self.dconv18 = BasicConv2d_norelu(in_channel=64, out_channel=64, kernel_size=3, padding=18, dilation=18)

        # self.conv_4 = nn.Conv2d(2 * 1024, 1024,kernel_size=3, padding=1)
        # self.conv_3 = nn.Conv2d(2 * 512, 512, kernel_size=3, padding=1)
        # self.conv_2 = nn.Conv2d(2 * 256, 256, kernel_size=3, padding=1)
        # self.conv_4 = nn.Conv2d(2 * 256, 256, kernel_size=3, padding=1)
        # self.conv_3 = nn.Conv2d(2 * 128, 128, kernel_size=3, padding=1)
        # self.conv_2 = nn.Conv2d(2 * 64, 64, kernel_size=3, padding=1)
        # self.conv = nn.Conv2d(64,nclass,kernel_size=3,padding=1)
        # self.relu = nn.ReLU(inplace=True)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.upsample4= nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        # self.sig = nn.Sigmoid()

        self.tconv1 = nn.ConvTranspose2d(inc, outc,kernel_size=3, stride=2, padding=1,output_padding=1, bias=False)
        self.tconv_end = nn.ConvTranspose2d(outc, outc, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x1,x2):
        x2 = self.tconv1(x2)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        # print(x1.shape)
        # print(x2.shape)
        out = x1+x2
        if self.flag==1:
            out = self.dconv1(out)
        elif self.flag==2:
            out = self.dconv2(out)
        else:
            out = self.dconv4(out)
            out = self.tconv_end(out)
        return out





        # x5 = self.upsample2(x5)
        # dout5 = self.dconv1(x5)
        #
        # x4 = torch.cat((x4,dout5),dim=1)
        # x4 = self.conv_4(x4)
        #
        # x4 = self.upsample2(x4)
        # dout4 = self.dconv6(x4)
        #
        # x3 = torch.cat((x3,dout4),dim=1)
        # x3 = self.conv_3(x3)
        #
        # x3 = self.upsample2(x3)
        # dout3 = self.dconv12(x3)
        #
        # x2 = torch.cat((x2,dout3),dim=1)
        # x2 = self.conv_2(x2)
        # dout2 = self.dconv18(x2)
        #
        #
        # out = self.upsample4(dout2)
        # out = self.conv(out)
        # dout6 = self.dconv6(x)
        # dout6 = x + dout6
        # dout6 = self.relu(dout6)
        # dout12 = self.dconv12(dout6)
        # dout12 = dout6 + dout12
        # dout12 = self.relu(dout12)
        # dout18 = self.dconv18(dout12)
        # dout18 = dout12 + dout18
        # dout18 = self.relu(dout18)
        # dout24 = self.dconv24(dout18)
        # out = dout18 + dout24
        # # out = self.relu(out)
        # out = self.conv(out)
        # # out = self.sig(dout24)
        # return out


# BBSNet
class BBSNet(nn.Module):
    def __init__(self, channel=32,n_class=None):
        super(BBSNet, self).__init__()

        # Backbone model

        self.resnet = ResNet34('rgb')  #64 64 128 256 512
        self.resnet_depth = ResNet34('rgbd')


        #ACM
        # self.acm1 = acm(64)
        # self.acm2 = acm(64)
        # self.acm3 = acm(128)
        # self.acm4 = acm(256)
        # self.acm5 = acm(512)
        #融合
        self.fusions = nn.ModuleList([
            fusion(64),
            fusion(128),
            fusion(256),
            fusion(512)

        ])
        self.refines_r_5 = nn.ModuleList([
            Refine(256,512,k=2),
            # Refine(128,512,k=4),
            # Refine(64,512,k=8)
        ])
        self.refines_r_4 = nn.ModuleList([
            Refine(128, 256,k=2),
            # Refine(64, 256,k=4)

        ])
        self.refines_r_3 = nn.ModuleList([
            Refine(64, 128,k=2),

        ])
        self.refines_d_5 = nn.ModuleList([
            Refine(256, 512,k=2),
            # Refine(128, 512,k=4),
            # Refine(64, 512,k=8)
        ])
        self.refines_d_4 = nn.ModuleList([
            Refine(128, 256,k=2),
            # Refine(64, 256,k=4)

        ])
        self.refines_d_3 = nn.ModuleList([
            Refine(64, 128,k=2),

        ])

        # self.conv_layer4 = BasicConv2d(2*512,512,kernel_size=3,padding=1)

        # self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
# #layer1_fusion细化conv1
#         self.conv1 = nn.Conv2d(2048*2,1024,kernel_size=3,padding=1)
#         self.conv2 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
#
#         self.bconv5 = BasicConv2d(in_channel=2048,out_channel=1024,kernel_size=3,padding=1)
#         self.bconv4 = BasicConv2d(in_channel=1024, out_channel=512, kernel_size=3, padding=1)
#         self.bconv3 = BasicConv2d(in_channel=512, out_channel=256, kernel_size=3, padding=1)
#         self.bconv2 = BasicConv2d(in_channel=256, out_channel=64, kernel_size=3, padding=1)
#         self.bconv1 = BasicConv2d(in_channel=64, out_channel=n_class, kernel_size=3, padding=1)
#
#         self.conv_end = nn.Conv2d(64,n_class,kernel_size=1,padding=0)

        # self.sgs = nn.ModuleList([
        #     SG(256,512,flag=1,in_plane=256),
        #     SG(128,256,flag=2,in_plane=128),
        #     SG(64,128,flag=3,in_plane=64),
        #     SG(64,64,c=False,flag=4,in_plane=64)
        # ])
        # #self.aspp = ASPP(num_classes=n_class)
        # #处理layer4_fusion
        # self.transconv = nn.ConvTranspose2d(512, 256, kernel_size=1, padding=0)
        # self.bn = nn.BatchNorm2d(256)
        #
        # 对每一层cat之后进行通道变换
        # self.conv_aux1 = nn.Conv2d(6,3,kernel_size=1,stride=1)
        # self.conv_aux2 = nn.Conv2d(64, n_class, kernel_size=1, stride=1)
        # self.conv_aux3 = nn.Conv2d(64, n_class, kernel_size=1, stride=1)
        # self.conv_aux4 = nn.Conv2d(64, n_class, kernel_size=1, stride=1)
        # self.decoder = serialaspp(nclass=n_class)
        self.decoder = nn.ModuleList([
            serialaspp(512,256,flag=1),
            serialaspp(256,128,flag=2),
            serialaspp(128,64,flag=3)
        ])

        self.conv_end = nn.Conv2d(64,n_class,kernel_size=1,padding=0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_aux1 = nn.Conv2d(256,n_class,kernel_size=1,padding=0)
        self.conv_aux2 = nn.Conv2d(128, n_class, kernel_size=1, padding=0)
        self.conv_aux3 = nn.Conv2d(64, n_class, kernel_size=1, padding=0)

        #加载预训练
        if self.training:
            self.initialize_weights()

    def forward(self, x, x_depth):
        x_depth = x_depth[:, :1, ...]
        #conv1  64 ,1/4
        x1 = self.resnet.conv1(x)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)

        x1 = self.resnet.maxpool(x1)
        #h,w = x1.size()[2:]
        x_depth1 = self.resnet_depth.conv1(x_depth)
        x_depth1 = self.resnet_depth.bn1(x_depth1)
        x_depth1 = self.resnet_depth.relu(x_depth1)

        x_depth1 = self.resnet_depth.maxpool(x_depth1)

        #layer1  256 1/4

        x2 = self.resnet.layer1(x1)
        x_depth2 = self.resnet_depth.layer1(x_depth1)

        #layer2  512  1/8
        x3 = self.resnet.layer2(x2)
        x_depth3 = self.resnet_depth.layer2(x_depth2)

        #layer3 1024 1/16

        x4 = self.resnet.layer3_1(x3)
        x_depth4 = self.resnet_depth.layer3_1(x_depth3)


        #layer4 2048 1/32

        x5 = self.resnet.layer4_1(x4)
        x_depth5 = self.resnet_depth.layer4_1(x_depth4)

        fuse5 = self.fusions[3](x5,x_depth5)
        x4 = self.refines_r_5[0](x4,fuse5)
        # x3 = self.refines_r_5[1](x3,fuse5)
        # x2 = self.refines_r_5[2](x2,fuse5)
        x_depth4 = self.refines_d_5[0](x_depth4,fuse5)
        # x_depth3 = self.refines_d_5[1](x_depth3, fuse5)
        # x_depth2 = self.refines_d_5[2](x_depth2, fuse5)
        fuse4 = self.fusions[2](x4,x_depth4)
        x3 = self.refines_r_4[0](x3, fuse4)
        # x2 = self.refines_r_4[1](x2, fuse4)
        x_depth3 = self.refines_d_4[0](x_depth3, fuse4)
        # x_depth2 = self.refines_d_4[1](x_depth2, fuse4)
        fuse3 = self.fusions[1](x3,x_depth3)
        x2 = self.refines_r_3[0](x2,fuse3)
        x_depth2 = self.refines_d_3[0](x_depth2,fuse3)
        fuse2 = self.fusions[0](x2,x_depth2)

        out45 = self.decoder[0](fuse4,fuse5) #256
        out43 = self.decoder[1](fuse3,out45)  #128
        out32 = self.decoder[2](fuse2,out43)  #64
        out = self.upsample2(out32)
        out = self.conv_end(out)
        a_out1 = self.conv_aux1(out45)
        a_out2 = self.conv_aux2(out43)
        a_out3 = self.conv_aux3(out32)
        # out = self.decoder(fuse2,fuse3,fuse4,fuse5)
        if self.training:
            return a_out1, a_out2, a_out3, out
        else:
            return out




    # initialize the weights
    def initialize_weights(self):

        #pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        res34 = models.resnet34(pretrained=True)
        pretrained_dict = res34.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)

        all_params = {}
        for k, v in self.resnet_depth.state_dict().items():
            if k == 'conv1.weight':
                all_params[k] = torch.nn.init.normal_(v, mean=0, std=1)
            elif k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet_depth.state_dict().keys())
        self.resnet_depth.load_state_dict(all_params)

if __name__ == '__main__':
    x = V(t.randn(2,3,480,640))
    y = V(t.randn(2,3,480,640))
    net = BBSNet(n_class=41)
    net1= net(x,y)
    print(net1.shape)


    # from torchsummary import summary
    # model = BBSNet(n_class=41)
    # model = model.cuda()
    # summary(model, input_size=[(3, 480, 640),(3,480,640)],batch_size=6)