import os
import utils
import dataset
import argparse
import torch
import time
import solver
import models
import test
import config as cfg
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(is_dist,start_epoch,local_rank,target=1,stack=1,style="Regression"):
    if(torch.cuda.device_count()==0):
        device=torch.device("cpu")
    else:
        device=torch.device("cuda:"+str(local_rank))
    if(local_rank==0):
        writer = SummaryWriter(log_dir="runs/s{}_target{}_rms".format(stack,target))

    KYDataset=dataset.KYDataset(is_train=True,stack=stack,target=target)
    dataloader=dataset.make_dataLoader(KYDataset,cfg.samples_per_gpu,is_dist)

    model=getattr(models, style)(is_train=True)
    if(start_epoch>1):
        utils.load_model(model,start_epoch-1)
    model=model.to(device)

    meters={"loss":utils.AverageMeter(),"time":utils.AverageMeter()}

    if(is_dist):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank, ], output_device=local_rank,
                                                          broadcast_buffers=False, find_unused_parameters=True)
    optimizer=solver.make_optimizer(model)
    model.train()

    for epoch in range(1,cfg.max_epochs+1):

        if is_dist:
            dataloader.sampler.set_epoch(epoch)

        end_time = time.time()

        for iteration,datas in enumerate(dataloader,1):

            x = datas[0].to(device)
            y = datas[1].to(device)
            loss, output=model(x,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            performance=utils.compute_performance(style,output,y)

            meters["loss"].update(loss.item())
            meters["time"].update(time.time()-end_time)
            end_time = time.time()

            if(local_rank==0):
                writer.add_scalar("loss/train",loss,(epoch-1)*len(dataloader)+iteration)
                for k in performance:
                    writer.add_scalar(k,performance[k].item(),(epoch-1)*len(dataloader)+iteration)

            if (iteration % 50 == 0):
                if (local_rank == 0):
                    res = [
                        "Epoch: [%d/%d]" % (epoch, cfg.max_epochs),
                        "Iter: [%d/%d]" % (iteration, len(dataloader)),
                        "lr: %.6f" % (optimizer.param_groups[0]["lr"])
                    ]

                    for k, v in meters.items():
                        message = "%.4f (%.4f)" % (v.val, v.avg)
                        res.append(k + ": " + message)

                    res = "\t".join(res)
                    print(res)

                for v in meters.values():
                    v.reset()

        if (local_rank == 0):
            utils.save_model(model, epoch)
            time.sleep(5)
            # performance=test.test(epoch,stack=stack,target=target)
            # for k in performance:
            #     writer.add_scalar(k, performance[k].item(), epoch)
        if (is_dist):
            utils.synchronize()

def main():
    parser = argparse.ArgumentParser(description="price predict")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--start_epoch", type=int, default=1)
    parser.add_argument("--target",type=int,default=1)
    parser.add_argument("--style",type=str,default="Regression")
    parser.add_argument("--dist", action="store_true", default=False)


    args = parser.parse_args()
    if (args.dist):
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        utils.synchronize()

    utils.init_seeds(0)
    train(args.dist, args.start_epoch, args.local_rank)

if __name__=="__main__":
    main()