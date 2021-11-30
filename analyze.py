import utils
import torch
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter

# epoch=1
#
# result=pd.read_csv("results/result_{}.csv".format(epoch))
# truth=torch.tensor(result["7"].values)
# predict=torch.tensor(result.drop(["7",],axis=1).values)
#
# truth_label=utils.encode(truth)
# predict_label=torch.argmax(predict,dim=1)
#
# m=confusion_matrix(truth_label,predict_label)
# print(torch.unique(predict_label))
# print(torch.unique(truth_label))
# print(m)
# print("done")

a=torch.randn(size=[2,2])
print(a)
a=a.permute(1,0)
print(a)
print(a.is_contiguous())
