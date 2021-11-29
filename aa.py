import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("result.csv")
predict=data["predict"].values
truth=data["truth"].values

max=predict.max()
min=predict.min()
print("max in predict:",max)
print("min in predict:",min)

index_pos_predict=predict>0
index_neg_predict=predict<0

index_pos_truth=truth>0
index_neg_truth=truth<0

index_0_truth=truth==0
index_0_truth=np.nonzero(index_0_truth)

index_pos = index_pos_predict*index_pos_truth
index_neg = index_neg_predict*index_neg_truth

print("both pos:", index_pos.sum())
print("both neg:", index_neg.sum())

print("total 0 of test:", index_0_truth.sum())

print("total test:",len(data))

times=np.abs(truth)/np.abs(predict)
print("mean times:",times.mean())

# plt.hist(predict,bins=200)
# plt.show()
#
# range_truth=(truth<0.002) * (truth>-0.002)
# range_truth=truth[range_truth]
# plt.hist(range_truth,bins=200)
# plt.show()