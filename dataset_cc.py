import numpy as np

def three_sigma(column):
    rule=(column.mean()-0.1*column.std() < column) & (column.mean()+0.1*column.std()>column)
    index=np.arange(column.shape[0])[rule]
    return index

def delete_outlier(data,exclude):
    out_index=[]
    nums=[]
    columns=[]
    for i in range(51):
        if(i in exclude):
            continue
        columns.append("feature"+str(i))
    columns.append("label_y")
    for c in columns:
        index=three_sigma(data[c])
        out_index+=index.tolist()
        nums.append(len(index.tolist()))

    delete_ = list(set(out_index))
    print("outlier sample:",len(delete_))
    print("outlier ratio:",len(delete_)/len(data))
    data.drop(index=delete_,inplace=True)
    return data,nums

def split(data,train_ratio=0.9):
    # 对数据进行划分并且进行归一化
    label=data["label_y"]
    data=data.drop(["Time","label_y"],axis=1)

    data=(data-data.mean())/data.std()

    train_len=int(len(data)*train_ratio)
    train_feature=data.iloc[:train_len,:]
    test_feature=data.iloc[train_len:,:]
    test_feature=test_feature.reset_index(drop=True)

    train_label=label[:train_len]
    test_label=label[train_len:]
    test_label=test_label.reset_index(drop=True)

    return train_label,train_feature,test_label,test_feature

