import torch
import torch.nn as nn
import torch.nn.functional as F
class Block(nn.Module):
    def __init__(self,inchannel,interchannel,outchannel):
        self.conv1 = nn.Conv2d(inchannel,interchannel,kernel_size=1, groups=32)
        self.bn1 = nn.BatchNorm2d(interchannel)
        self.conv2 = nn.Conv2d(interchannel,interchannel,kernel_size=3,padding=1,groups=32)
        self.bn2 = nn.BatchNorm2d(interchannel)
        self.conv3 = nn.Conv2d(interchannel,outchannel,kernel_sizie=1, groups=32)
        self.bn3 = nn.BatchNorm2d(outchannel)
    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = F.relu(self.bn2(out))
        out = self.conv3(out)
        out = F.relu(self.bn3(out))
        out = F.relu(torch.cat(x,out))
        return out

class ResNeXt(nn.Module):
    def __init__(self):
        super(ResNeXt,self).__init__()
        self.conv1 = nn.Conv2d(3,64,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer = self._make_layer_([[64,128,256],[256,256,512],[512,512,1024]])
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
        self.avgpool = nn.AvgPool2d(4,1,ceil_mode=True)
        self.linear = nn.Linear(1024,100)
    def _make_layer_(self,params):
        list = []
        for j in range(3):
            for i in range(3):
                list.append(Block(params[j][0],params[j][1],params[j][2]))
        return nn.Sequential(*list)
    def forward(self,x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.layer(out)
        out = self.avgpool(out)
        out.view(out.size(0),-1)
        out = self.linear(out)
        return out