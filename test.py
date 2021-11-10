import torch
import model
import utils
import argparse
import dataset
from tqdm import tqdm

def test(epoch,sequence_length,target=1):
    if (torch.cuda.device_count() == 0):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    test_Dataset = dataset.KYDataset(is_train=False,sequence_length=sequence_length)
    test_dataloader = dataset.make_dataLoader(test_Dataset,batchsize=1,is_dist=False,is_train=False)
    gru = model.GRU(is_train=False)
    utils.load_model(gru, epoch)

    gru = gru.to(device)
    total_mse=0
    i=1

    numerator = 0
    denominator = 0
    with torch.no_grad():
        with tqdm(total=len(test_Dataset),desc=f"Epoch #{epoch}") as t:
            for datas in test_dataloader:
                x = datas[0].to(device)
                y = datas[1].to(device)
                predict = gru(x, y)

                predict=predict.view(-1)
                y=y.view(-1)

                cur_numerator = torch.sum(utils.compute_numerator(predict[-1], y[-1]))
                cur_denominator = torch.sum(utils.compute_denominator(y[:, -1], test_Dataset.label_mean))
                numerator = (numerator * (i - 1)  + cur_numerator) /i
                denominator = (denominator * (i - 1) + cur_denominator) / i
                i+=1
                t.update(1)

    return 1-numerator/denominator

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="price predict")
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--target", type=int, default=1)
    parser.add_argument("--sequence_length", type=int, default=100)

    args = parser.parse_args()

    test(args.epoch,args.sequence_length,args.target)