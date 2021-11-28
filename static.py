import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

data_path="data/train_1.csv"
data=pd.read_csv(data_path)

data=data["target1"].values

def encode(values):
    values=np.clip(values,-0.015,0.015-(1e-6))
    label=(values+0.015)/0.001
    label=label.astype(np.int)
    count=Counter(label)

    b=values==0
    b=np.sum(b)

    index=label==15
    values=values[index]
    print(np.max(values))
    print(np.min(values))

    x=count.keys()
    x=sorted(x)
    y=[count[_] for _ in x]

    fig = plt.figure()
    plt.bar(x,y)
    plt.savefig("target1.png")

encode(data)
index=data>0
c=np.sum(index)
print(c)