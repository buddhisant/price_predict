import torch
from torch import nn


m=nn.LayerNorm([2,1,2])
n=nn.BatchNorm1d(3)
p=nn.GroupNorm(num_groups=2,num_channels=10)
t=nn.InstanceNorm1d(num_features=10)

inp=[[[[1,1]],[[2,3]]]]
inp=torch.tensor(inp,dtype=torch.float)
out=m(inp)

print("done")
