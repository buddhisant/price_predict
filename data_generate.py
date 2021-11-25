import os
import utils
import numpy as np
import pandas as pd
import datetime as dt

utils.mkdir("data")

root_path="/home/buddhisant/data/datafeatures"
name1="000001.sz_intern_20200101-20210329.h5"
name2="000002.sz_intern_20200101-20210329.h5"
name3="000063.sz_intern_20200101-20210329.h5"

raw_df=pd.read_hdf(os.path.join(root_path,name3),mode="r")
raw_df=raw_df.sort_index()

raw_df=raw_df[raw_df["ChgToPreClose"].between(-0.07,0.08)]
raw_df[np.isinf(raw_df)]=np.nan
df=raw_df.dropna(axis=0,how="any")

print(df.shape)

split_time='20210201'
train_df=df[df["DataCreatedTime"]<=dt.datetime.strptime(split_time,'%Y%m%d')]
test_df=df[df["DataCreatedTime"]>dt.datetime.strptime(split_time,'%Y%m%d')]

train_df=train_df.drop(["DataCreatedTime"],axis=1)
test_df=test_df.drop(["DataCreatedTime"],axis=1)

train_df.to_csv("data/train_63.csv")
test_df.to_csv("data/test_63.csv")
print(test_df.shape)

