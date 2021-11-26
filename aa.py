import torch
from torch import nn
import pandas as pd

data=pd.read_csv("data/train_1.csv",index_col=0)
max_l=data["target1"].max()
min_l=data["target1"].min()

print(max_l)
print(min_l)
