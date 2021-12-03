import math
import torch
import utils
import config as cfg

from torch import nn
from resnet import resNet
from loss import ClassLoss

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

class Classification(nn.Module):
    def __init__(self,is_train=True):
        super(Classification, self).__init__()
        self.is_train=is_train

        self.conv1=nn.Conv1d(1,200,kernel_size=5,padding=2)
        self.ln1=nn.LayerNorm([200,175])
        self.bn1=nn.BatchNorm1d(200)
        self.conv2=nn.Conv1d(200,150,kernel_size=5,padding=2)
        self.ln2 = nn.LayerNorm([150,88])
        self.bn2 = nn.BatchNorm1d(150)
        self.conv3=nn.Conv1d(150,100,kernel_size=5,padding=2)
        self.ln3 = nn.LayerNorm([100,45])
        self.bn3 = nn.BatchNorm1d(100)
        self.conv4=nn.Conv1d(100,60,kernel_size=3,padding=1)
        self.ln4 = nn.LayerNorm([60,23])
        self.bn4 = nn.BatchNorm1d(60)
        self.conv5=nn.Conv1d(60,40,kernel_size=3,padding=1)
        self.ln5 = nn.LayerNorm([40,12])
        self.bn5 = nn.BatchNorm1d(40)

        self.linear=nn.Linear(480,7)

        self.maxpool=nn.MaxPool1d(kernel_size=2,padding=1)
        self.tanh=nn.Tanh()
        self.act=self.tanh

        # self.relu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.dropout=nn.Dropout(p=0.1,inplace=False)
        self.loss=ClassLoss()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if(isinstance(m, torch.nn.Conv1d)):
                torch.nn.init.orthogonal_(m.weight)
                if (hasattr(m, "bias") and m.bias is not None):
                    torch.nn.init.constant_(m.bias, 0)
            elif (isinstance(m, torch.nn.Linear)):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1 - cfg.class_prior_prob) / cfg.class_prior_prob)
        torch.nn.init.constant_(self.linear.bias,bias_value)

    def forward(self,x,y):
        x = self.act(self.dropout(self.conv1(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv2(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv3(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv4(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv5(x)))

        x=x.flatten(1)
        predict=self.linear(x)
        predict.sigmoid_()

        if self.is_train:
            loss = self.loss(predict,y)
            return loss, predict

        return predict

class Classification1(nn.Module):
    def __init__(self,is_train=True):
        super(Classification1, self).__init__()
        self.is_train=is_train

        self.conv1=nn.Conv1d(1,200,kernel_size=5,padding=2)
        self.ln1=nn.LayerNorm([200,175])
        self.bn1=nn.BatchNorm1d(200)
        self.conv2=nn.Conv1d(200,150,kernel_size=5,padding=2)
        self.ln2 = nn.LayerNorm([150,88])
        self.bn2 = nn.BatchNorm1d(150)
        self.conv3=nn.Conv1d(150,100,kernel_size=5,padding=2)
        self.ln3 = nn.LayerNorm([100,45])
        self.bn3 = nn.BatchNorm1d(100)
        self.conv4=nn.Conv1d(100,60,kernel_size=3,padding=1)
        self.ln4 = nn.LayerNorm([60,23])
        self.bn4 = nn.BatchNorm1d(60)
        self.conv5=nn.Conv1d(60,40,kernel_size=3,padding=1)
        self.ln5 = nn.LayerNorm([40,12])
        self.bn5 = nn.BatchNorm1d(40)

        self.linear=nn.Linear(40,30)

        self.maxpool=nn.MaxPool1d(kernel_size=2,padding=1)
        self.tanh=nn.Tanh()
        self.act=self.tanh

        # self.relu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.dropout=nn.Dropout(p=0.1,inplace=False)
        self.loss=ClassLoss()

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if(isinstance(m, torch.nn.Conv1d)):
                torch.nn.init.orthogonal_(m.weight)
                if (hasattr(m, "bias") and m.bias is not None):
                    torch.nn.init.constant_(m.bias, 0)
            elif (isinstance(m, torch.nn.Linear)):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

        bias_value = -math.log((1 - cfg.class_prior_prob) / cfg.class_prior_prob)
        torch.nn.init.constant_(self.linear.bias,bias_value)

    def forward(self,x,y):
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
        predict=self.linear(x)
        predict.sigmoid_()

        if self.is_train:
            loss = self.loss(predict,y)
            return loss, predict

        return predict

class Regression1(nn.Module):
    def __init__(self,is_train=True):
        super(Regression1, self).__init__()
        self.is_train=is_train

        self.fc1=nn.Linear(175,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,128)
        self.fc4=nn.Linear(128,64)
        self.fc5=nn.Linear(64,32)
        self.fc6=nn.Linear(32,1)

        self.linear=nn.Linear(40,1)

        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.act=self.tanh

        self.dropout=nn.Dropout(p=0.1,inplace=False)

        for m in self.modules():
            if(isinstance(m, torch.nn.Conv1d)):
                torch.nn.init.orthogonal_(m.weight)
                if (hasattr(m, "bias") and m.bias is not None):
                    torch.nn.init.constant_(m.bias, 0)
            elif (isinstance(m, torch.nn.Linear)):
                torch.nn.init.orthogonal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self,x,label):
        x = self.act(self.dropout(self.fc1(x)))
        x = self.act(self.dropout(self.fc2(x)))
        x = self.act(self.dropout(self.fc3(x)))
        x = self.act(self.dropout(self.fc4(x)))
        x = self.act(self.dropout(self.fc5(x)))
        y = self.fc6(x)

        y=y.view(-1)

        if self.is_train:
            label=label*cfg.scale
            loss = torch.log(torch.cosh(y-label))
            loss = loss.sum()
            return loss, y/cfg.scale

        return y/cfg.scale

class Regression(nn.Module):
    def __init__(self,is_train=True):
        super(Regression, self).__init__()
        self.is_train=is_train

        self.conv1=nn.Conv1d(1,200,kernel_size=5,padding=2)
        self.ln1=nn.LayerNorm([200,175])
        self.bn1=nn.BatchNorm1d(200)
        self.conv2=nn.Conv1d(200,150,kernel_size=5,padding=2)
        self.ln2 = nn.LayerNorm([150,88])
        self.bn2 = nn.BatchNorm1d(150)
        self.conv3=nn.Conv1d(150,100,kernel_size=5,padding=2)
        self.ln3 = nn.LayerNorm([100,45])
        self.bn3 = nn.BatchNorm1d(100)
        self.conv4=nn.Conv1d(100,60,kernel_size=3,padding=1)
        self.ln4 = nn.LayerNorm([60,23])
        self.bn4 = nn.BatchNorm1d(60)
        self.conv5=nn.Conv1d(60,40,kernel_size=3,padding=1)
        self.ln5 = nn.LayerNorm([40,12])
        self.bn5 = nn.BatchNorm1d(40)

        self.linear=nn.Linear(40,1)

        self.maxpool=nn.MaxPool1d(kernel_size=2,padding=1)
        self.maxpool0=nn.MaxPool1d(kernel_size=2)
        self.avgpool=nn.AvgPool1d(kernel_size=2,)
        self.tanh=nn.Tanh()
        self.relu=nn.ReLU()
        self.act=self.tanh

        # self.relu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.dropout=nn.Dropout(p=0.1,inplace=False)

        self.smoothL1=torch.nn.SmoothL1Loss()
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
        # [128,1,175]
        x = self.act(self.dropout(self.conv1(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv2(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv3(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv4(x)))
        x = self.maxpool(x)

        x = self.act(self.dropout(self.conv5(x)))
        # [128,40,12]

        x=x.mean(axis=-1)
        # [128,40]
        y=self.linear(x)

        y=y.view(-1)

        if self.is_train:
            label=label*cfg.scale
            loss=self.smoothL1(label,y)
            # loss = torch.log(torch.cosh(y-label))
            loss = loss.sum()
            return loss, y/cfg.scale

        return y/cfg.scale


if __name__=="__main__":
    inp=torch.randn(size=[128,1,175]).cuda()
    m=Regression().cuda()
    out=m(inp,0)

