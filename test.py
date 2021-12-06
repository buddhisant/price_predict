import torch
import models
import utils
import argparse
import dataset

import numpy as np
import pandas as pd

from tqdm import tqdm

def test(epoch,Dataset,style="Regression",dataset_name="test"):
    if (torch.cuda.device_count() == 0):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    test_dataloader = dataset.make_dataLoader(Dataset,batchsize=128,is_dist=False,is_train=False)

    model = getattr(models, style)(is_train=False)
    utils.load_model(model, epoch)
    model = model.to(device)
    model.eval()
    predicts=[]
    ys=[]
    with torch.no_grad():
        with tqdm(total=len(test_dataloader),desc=f"Epoch #{epoch}") as t:
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

    performace=utils.compute_performance(style, predicts, ys, mode=dataset_name)

    predicts=predicts.cpu().numpy()
    predicts=predicts.reshape([-1,1])
    ys=ys.cpu().numpy()
    ys=ys.reshape([-1,1])

    predicts=np.concatenate([predicts,ys],axis=1)

    predicts=pd.DataFrame(predicts)

    utils.mkdir("results")
    predicts.to_csv("results/{}_{}.csv".format(dataset_name,epoch), index=False)
    return performace

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="price predict")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)

    args = parser.parse_args()

    test_Dataset = dataset.KYDataset(is_train=False, target=args.target, stack=1)
    test(args.epoch,test_Dataset)