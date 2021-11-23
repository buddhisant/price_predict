import os
import torch
import random
import pandas as pd
from torch import nn
from torch.utils.data import Dataset,distributed,DataLoader

root_path="/home/buddhisant/data/datafeatures"
name1="000001.sz_intern_20200101-20210329.h5"

class KYDataset(Dataset):
    non_factor=["ChgToPreClose","Match","AskPrice1","BidPrice1","Volume","Turnover"]

    def __init__(self,is_train=True,target=1):
        self.is_train=is_train

        df = pd.read_csv(os.path.join("data", "train.csv"), index_col=0)
        df = df.drop(["target1","target2","target3"]+self.non_factor,axis=1)
        df = df.values
        df = torch.tensor(df,dtype=torch.float)
        self.mean=df.mean(axis=0).view(1,-1)
        self.std=df.std(axis=0).view(1,-1)

        if(self.is_train):
            df=pd.read_csv(os.path.join("data","train.csv"),index_col=0)
        else:
            df=pd.read_csv(os.path.join("data","test.csv"),index_col=0)

        self.target_name="target"+str(target)
        drop_targets=list(set([1,2,3])-set([target]))
        drop_targets=["target"+str(i) for i in drop_targets]
        self.data=df.drop(drop_targets+self.non_factor,axis=1)

        self.label_mean=self.mean_label()

    def __len__(self):
        return len(self.data)

    def mean_label(self):
        label=self.data[self.target_name]
        label=label.values
        label=torch.tensor(label,dtype=torch.float)
        return label.mean()

    def __getitem__(self, idx):
        data=self.data.iloc[idx]
        label=data[self.target_name]
        data = data.drop(self.target_name)

        data = data.values
        data = torch.tensor(data,dtype=torch.float)
        # label=label.values
        label=torch.tensor(label,dtype=torch.float)

        data=(data-self.mean)/self.std

        return data,label

def make_dataLoader(dataset,batchsize,is_dist,is_train=True):
    sampler=distributed.DistributedSampler(dataset) if is_dist else None
    dataloader=DataLoader(dataset,batch_size=batchsize,sampler=sampler,shuffle=is_train)
    return dataloader

