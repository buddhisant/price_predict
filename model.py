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

        self.conv1=nn.Conv1d(1,200,kernel_size=5,padding=2)
        self.conv2=nn.Conv1d(200,150,kernel_size=5,padding=2)
        self.conv3=nn.Conv1d(150,100,kernel_size=5,padding=2)
        self.conv4=nn.Conv1d(100,60,kernel_size=3,padding=1)
        self.conv5=nn.Conv1d(60,40,kernel_size=3,padding=1)
        self.linear=nn.Linear(40,1)

        self.maxpool=nn.MaxPool1d(kernel_size=2,padding=1)
        self.maxpool0=nn.MaxPool1d(kernel_size=2)
        self.avgpool=nn.AvgPool1d(kernel_size=2,)
        self.tanh=nn.Tanh()
        self.act=self.tanh

        self.relu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.dropout=nn.Dropout(p=0.1,inplace=False)

        self.mse=torch.nn.MSELoss()

        for m in self.modules():
            if(isinstance(m, torch.nn.Conv1d)):
                torch.nn.init.orthogonal_(m.weight)
                if (hasattr(m, "bias") and m.bias is not None):
                    torch.nn.init.constant_(m.bias, 0)
            elif (isinstance(m, torch.nn.Linear)):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self,x,label):
        x = self.act(self.dropout(self.conv1(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv2(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv3(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv4(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv5(x)))

        x=x.mean(axis=-1)
        x=self.linear(x)
        y=self.act(x)

        y=y.view(-1)

        if self.is_train:
            label=label*cfg.scale
            loss=self.mse(label,y)
            return loss, y/cfg.scale

        return y/cfg.scale

