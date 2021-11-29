import torch
import models
import utils
import argparse
import dataset

import numpy as np
import pandas as pd

from tqdm import tqdm

def test(epoch,target=1,stack=1,style="Classification"):
    if (torch.cuda.device_count() == 0):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    test_Dataset = dataset.KYDataset(is_train=False,target=target,stack=stack)
    test_dataloader = dataset.make_dataLoader(test_Dataset,batchsize=1,is_dist=False,is_train=False)

    for epoch in range(1,epoch+1):
        model = getattr(models, style)(is_train=False)
        utils.load_model(model, epoch)
        model = model.to(device)
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

        predicts=predicts.cpu().numpy()
        ys=ys.cpu().numpy()
        ys=ys.reshape([-1,1])

        predicts=np.concatenate([predicts,ys],axis=1)

        predicts=pd.DataFrame(predicts)
        predicts.to_csv("results/result_{}.csv".format(epoch), index=False)
    return performace

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="price predict")
    parser.add_argument("--epoch", type=int, default=6)
    parser.add_argument("--target", type=int, default=1)

    args = parser.parse_args()

    test(args.epoch,args.target)