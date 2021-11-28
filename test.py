import torch
import p_model
import utils
import argparse
import dataset

import pandas as pd

from tqdm import tqdm

def test(epoch,target=1,stack=1,style="Regression"):
    if (torch.cuda.device_count() == 0):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    test_Dataset = dataset.KYDataset(is_train=False,target=target,stack=stack)
    test_dataloader = dataset.make_dataLoader(test_Dataset,batchsize=1,is_dist=False,is_train=False)
    model = getattr(p_model,style)(is_train=False)
    utils.load_model(model, epoch)

    model = model.to(device)

    numerator = 0
    denominator = 0
    count=0
    model.eval()
    predicts=[]
    ys=[]
    with torch.no_grad():
        with tqdm(total=len(test_Dataset),desc=f"Epoch #{epoch}") as t:
            for datas in test_dataloader:
                x = datas[0].to(device)
                # x = x.permute(0, 2, 1).contiguous()
                y = datas[1].to(device)
                predict = model(x, y)

                predicts.append(predict)
                ys.append(y)

                t.update(1)
    predicts=torch.cat(predicts)
    ys=torch.cat(ys)

    performace=utils.compute_performance(style,predicts,ys,mode="test")

    predicts=predicts.cpu()
    ys=ys.cpu()

    result=pd.DataFrame({"predict":predicts,"truth":ys})
    result.to_csv("result.csv",index=False)
    return performace

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="price predict")
    parser.add_argument("--epoch", type=int, default=6)
    parser.add_argument("--target", type=int, default=1)

    args = parser.parse_args()

    test(args.epoch,args.target)