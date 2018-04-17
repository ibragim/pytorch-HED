from torch import nn
from torch.nn import functional as F
import torch
from config import cfg
import numpy as np
import math
class BaseVgg(nn.Module):
    def __init__(self):
        super(BaseVgg,self).__init__()
        self.conv1_1=nn.Conv2d(3,64,3,1,1)
        self.conv1_2=nn.Conv2d(64,64,3,1,1)

        self.conv2_1=nn.Conv2d(64,128,3,1,1)
        self.conv2_2=nn.Conv2d(128,128,3,1,1)

        self.conv3_1=nn.Conv2d(128,256,3,1,1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

        self.conv5_1=nn.Conv2d(512, 512, 3, 1, 2,dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 2,dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 2,dilation=2)

    def forward(self, x):
        x=self.conv1_1(x)
        x=F.relu(x,inplace=True)
        x=self.conv1_2(x)
        c1=F.relu(x)

        x=F.max_pool2d(c1,2,2,padding=0)
        x=self.conv2_1(x)
        x=F.relu(x)
        x=self.conv2_2(x)
        c2=F.relu(x)

        x=F.max_pool2d(c2,2,2,padding=0)
        x=self.conv3_1(x)
        x=F.relu(x)
        x=self.conv3_2(x)
        x=F.relu(x)
        x=self.conv3_3(x)
        c3=F.relu(x)

        x=F.max_pool2d(c3,2,2,padding=0)
        x=self.conv4_1(x)
        x=F.relu(x)
        x=self.conv4_2(x)
        x=F.relu(x)
        x=self.conv4_3(x)
        c4=F.relu(x)

        x=F.max_pool2d(c4,2,1,padding=0)
        x=self.conv5_1(x)
        x=F.relu(x)
        x=self.conv5_2(x)
        x=F.relu(x)
        x=self.conv5_3(x)
        c5=F.relu(x)

        return c1,c2,c3,c4,c5

class HED(nn.Module):
    def __init__(self):
        super(HED,self).__init__()

        self.upsame2=nn.ConvTranspose2d(1,1,4,2,padding=0,bias=False)
        self.upsame4=nn.ConvTranspose2d(1,1,8,4,padding=0,bias=False)
        self.upsame8 = nn.ConvTranspose2d(1,1,16,8,padding=0,bias=False)
        self.upsame16 =nn.ConvTranspose2d(1,1,16,8,padding=0,bias=False)

        self.conv1=nn.Conv2d(64,1,1)
        self.conv2=nn.Conv2d(128,1,1)
        self.conv3=nn.Conv2d(256,1,1)
        self.conv4=nn.Conv2d(512,1,1)
        self.conv5=nn.Conv2d(512,1,1)
        self.convfuse=nn.Conv2d(5,1,1,bias=False)
        self.basevgg=BaseVgg()

    def forward(self, x):
        c1,c2,c3,c4,c5=self.basevgg(x)
        size=c1.size()[2:4]
        #self.upsame2 = nn.UpsamplingBilinear2d(size=size)
        #self.upsame4 = nn.UpsamplingBilinear2d(size=size)
        #self.upsame8 = nn.UpsamplingBilinear2d(size=size)
        #self.upsame16 = nn.UpsamplingBilinear2d(size=size)

        b1=self.conv1(c1)
        b2=self.upsame2(self.conv2(c2))
        b2=b2[:,:,1:1+size[0],1:1+size[1]]

        b3=self.upsame4(self.conv3(c3))
        b3 = b3[:, :, 2:2 + size[0], 2:2 + size[1]]

        b4=self.upsame8(self.conv4(c4))
        b4 = b4[:, :, 4:4 + size[0], 4:4 + size[1]]

        b5=self.upsame8(self.conv5(c5))


        bfuse=self.convfuse(torch.cat([b1,b2,b3,b4,b5],dim=1))

        b1=F.sigmoid(b1)
        b2 = F.sigmoid(b2)
        b3 = F.sigmoid(b3)
        b4 = F.sigmoid(b4)
        b5 = F.sigmoid(b5)
        bfuse = F.sigmoid(bfuse)
        return b1,b2,b3,b4,b5,bfuse

    def upsample_filt(self,size):
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

    def init_weights(self):
        """
        for m in self.modules():
            if(isinstance(m,nn.Conv2d)):
                nn.init.uniform(m.weight,0,cfg.conv_weight_normal)
                if(m.bias is not None):
                    m.bias.data.zero_()
        """
        nn.init.constant(self.convfuse.weight,cfg.conv_weight_normal)
        self.conv1.weight.data.zero_()
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()
        self.conv4.weight.data.zero_()
        self.conv4.bias.data.zero_()
        self.conv5.weight.data.zero_()
        self.conv5.bias.data.zero_()

        upsample_layers=[self.upsame2,self.upsame4,self.upsame8,self.upsame16]
        for m in upsample_layers:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return (1 - abs(og[0] - center) / factor) * \
                   (1 - abs(og[1] - center) / factor)

        for l in upsample_layers:
            m, k, h, w = l.weight.data.shape
            if m != k:
                print('input + output channels need to be the same')
            if h != w:
                print('filters need to be square')
            filt = upsample_filt(h)
            l.weight.data[0, 0, :, :] = torch.from_numpy(filt)
            for param in l.parameters():
                param.requires_grad=False



