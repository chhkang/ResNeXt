import torch
import torch.nn as nn
import torch.nn.functional as F
class Block(nn.Module):
    def __init__(self,inchannel,interchannel,outchannel,pool_stride):
        super(Block,self).__init__()
        self.conv1 = nn.Conv2d(inchannel,interchannel,kernel_size=1)
        self.bn1 = nn.BatchNorm2d(interchannel)
        self.conv2 = nn.Conv2d(interchannel,interchannel,kernel_size=3,stride=pool_stride,padding=1,groups=32)
        self.bn2 = nn.BatchNorm2d(interchannel)
        self.conv3 = nn.Conv2d(interchannel,outchannel,kernel_size=1)
        self.bn3 = nn.BatchNorm2d(outchannel)
        self.shortcut = nn.Sequential()
        self.shortcut.add_module('shortcut_conv',nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=pool_stride,
                                                           padding=0, bias=False))
        self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(outchannel))
    def forward(self,x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = F.relu(self.bn1(out), inplace=True)
        out = self.conv2(out)
        out = F.relu(self.bn2(out), inplace=True)
        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(residual+out, inplace=True)

class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer = self._make_layer_([[64,128,256],[256,256,512],[512,512,1024]])
        self.avgpool = nn.AvgPool2d(4,1,ceil_mode=True)
        self.linear = nn.Linear(1024, 100)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_layer_(self,params):
        list = []
        for param in params:
            list.append(Block(param[0],param[1],param[2],2))
            for i in range(2):
                list.append(Block(param[2],param[1],param[2],1))
        return nn.Sequential(*list)
    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.linear(out)
        return out