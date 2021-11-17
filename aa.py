import torch
import pandas as pd


a=torch.ones(size=[2,3,4])
b=a.sum(dim=2)
print(b.shape)
print(b)