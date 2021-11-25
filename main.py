import os
import csv
import numpy as np
from tqdm import tqdm
from dataset_cc import three_sigma

import pandas as pd
import matplotlib.pyplot as plt

# print(df.info())
# print(len(df))
# columns=df.columns.values
# print(columns)

df=pd.read_csv("data/train_1.csv", index_col=0)

df=df.reset_index(drop=True)

# print(df["fatcor_175"].describe())
# print(df.columns)

# 绘制每个特征及label的分布直方图
# features=["fatcor_"+str(i) for i in range(1,176)]
# features.append("target1")
# for f in tqdm(features):
#     fig=plt.figure()
#     plt.hist(df[f],bins=200)
#     plt.title(f)
#     plt.savefig(os.path.join("distribution",f+".png"))

f="target1"
dd=df[f]

index=three_sigma(dd)
dd=dd[index]

fig=plt.figure()
plt.hist(dd,bins=200)
plt.title(f)
plt.savefig(f+".png")
