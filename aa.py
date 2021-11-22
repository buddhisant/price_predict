import torch
import pandas as pd


a=torch.tensor([1,2,3,4,5])
b=torch.ones(size=[5,1])
c=a-b
print(c)