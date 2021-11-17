import torch
import config as cfg

from torch import nn
from resnet import resNet

class GRU(torch.nn.Module):
    def __init__(self,is_train,input_size=181,hidden_size=128):
        super(GRU, self).__init__()
        self.gru=torch.nn.GRU(input_size=input_size,hidden_size=hidden_size,batch_first=True)
        self.fc1=torch.nn.Linear(hidden_size,int(hidden_size/2))
        self.fc2=torch.nn.Linear(int(hidden_size/2),int(hidden_size/4))
        self.fc3=torch.nn.Linear(int(hidden_size/4),1)
        self.loss=torch.nn.MSELoss(reduction="sum")

        self.is_train=is_train
        self.initialize()

    def initialize(self):
        for name,parameter in self.named_parameters():
            if("weight" in name):
                torch.nn.init.normal_(parameter,mean=0,std=0.01)
            if("bias" in name):
                torch.nn.init.constant_(parameter,0.0)

    def forward(self,input,label):
        hn_output=self.gru(input)[0]

        output=self.fc1(hn_output)
        output.relu_()
        output=self.fc2(output)
        output.relu_()
        output=self.fc3(output)
        if(self.is_train):
            loss=self.loss(output.flatten(1),label)
            return loss,output

        else:
            return output


class Regression(nn.Module):
    def __init__(self,is_train=True):
        super(Regression, self).__init__()
        self.is_train=is_train

        self.resnet=resNet()

        self.fconv1=torch.nn.Conv1d(2048,cfg.fpn_channels,1,)
        self.fconv2=torch.nn.Conv1d(cfg.fpn_channels,cfg.fpn_channels,3,padding=1)
        self.conv1=torch.nn.Conv1d(cfg.fpn_channels,cfg.fpn_channels,3,padding=1,dilation=2)
        self.gn1 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv2=torch.nn.Conv1d(cfg.fpn_channels,cfg.fpn_channels,3,padding=1,dilation=2)
        self.gn2 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv3=torch.nn.Conv1d(cfg.fpn_channels,cfg.fpn_channels,3,padding=1,dilation=2)
        self.gn3 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)
        self.conv4=torch.nn.Conv1d(cfg.fpn_channels,cfg.fpn_channels,3,padding=1,dilation=2)
        self.gn4 = torch.nn.GroupNorm(num_channels=cfg.fpn_channels, num_groups=32)

        self.fc1 = torch.nn.Linear(cfg.fpn_channels,1)
        self.mse=torch.nn.MSELoss()

        for m in self.modules():
            if(isinstance(m, torch.nn.Conv1d)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if (hasattr(m, "bias") and m.bias is not None):
                    torch.nn.init.constant_(m.bias, 0)
            elif (isinstance(m, torch.nn.GroupNorm)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self,input,label):
        x=self.resnet(input)
        x=self.fconv1(x)
        x=self.fconv2(x)

        x=self.conv1(x)
        x=self.gn1(x)
        x.relu_()

        x = self.conv2(x)
        x = self.gn2(x)
        x.relu_()

        x = self.conv3(x)
        x = self.gn3(x)
        x.relu_()

        x = self.conv4(x)
        x = self.gn4(x)
        x.relu_()

        x=x.mean(dim=-1)
        y=self.fc1(x)
        y=y.view(-1)

        if self.is_train:
            label=label[:,-1]*cfg.scale
            loss=self.mse(label,y)
            return loss, y/cfg.scale

        return y/cfg.scale
