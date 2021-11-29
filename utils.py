import os
import torch
import torch
import config as cfg
import torch.distributed as dist

def encode(values):
    label = (values - cfg.max_decline) / cfg.interval + 1
    label = label.int()
    num_classes = int((cfg.max_increase - cfg.max_decline) / cfg.interval) + 2

    label[values == 0] = num_classes
    label[values < cfg.max_decline]=0
    label[values>cfg.max_increase]=num_classes-1
    return label

def decode(probability):
    start=cfg.max_decline+cfg.interval/2
    end=cfg.max_increase-cfg.interval/2
    num_interval=int((cfg.max_increase-cfg.max_decline)/cfg.interval)
    center_val=torch.linspace(start,end,num_interval,device=probability.device)
    center_val=center_val.view(1,-1)

    probability=probability.softmax(dim=1)
    predict=probability*center_val
    predict=predict.sum(dim=1)

    return predict

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def save_model(model,epoch):
    """保存训练好的模型，同时需要保存当前的epoch"""
    if(hasattr(model,"module")):
        model=model.module
    model_state_dict=model.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()
    checkpoint=dict(state_dict=model_state_dict,epoch=epoch)
    mkdir(cfg.archive_path)
    checkpoint_name=cfg.check_prefix+"_"+str(epoch)+".pth"
    checkpoint_path=os.path.join(cfg.archive_path,checkpoint_name)

    torch.save(checkpoint,checkpoint_path)

def load_model(model, epoch):
    """
    加载指定的checkpoint文件
    :param model:
    :param epoch:
    :return:
    """
    archive_path = os.path.join(cfg.archive_path, cfg.check_prefix+"_"+str(epoch)+".pth")
    check_point = torch.load(archive_path)
    state_dict = check_point["state_dict"]
    model.load_state_dict(state_dict)

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def synchronize():
    """启用分布式训练时，用于各个进程之间的同步"""
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def compute_numerator(y_pre,y_label):
    return (y_pre-y_label)**2

def compute_denominator(y_label):
    mean=y_label.mean()
    return (y_label-mean)**2

def compute_r2(preidict,y):
    numerator=compute_numerator(preidict,y).sum()
    denominator=compute_denominator(y).sum()
    return 1-numerator/denominator

def compute_p(label_predict,y):
    label_gt=encode(y)
    equal=(label_predict==label_gt).int()
    return equal.sum()/len(equal)

def compute_performance(style,output,y,mode="train"):
    performance={}
    num_classes=output.size(-1)
    if(style=="Classification"):
        # value_predict = decode(output)
        label_predict = torch.argmax(output, dim=1)
        probability=output[range(len(output)),label_predict]

        # r2 = compute_r2(value_predict, y)
        p = compute_p(label_predict, y)
        # performance[mode+"/r2"]=r2
        performance[mode+"/p"]=p

    if(style=="Regression"):
        output=output.view(-1)
        r2=compute_r2(output,y)
        performance[mode+"/r2"]=r2

    return performance

def compute_ap(predict_right, truth_right, score):
    index=torch.argsort(score,descending=True)
    predict_right=predict_right[index].float()
    truth_right=truth_right[index].float()
    total_pos = truth_right.sum()

    p_denominator=torch.range(1,len(predict_right)+1,dtype=torch.float)
    p_denominator=torch.cumsum(p_denominator,dim=0)

    p_numerator=torch.cumsum(predict_right,dim=0)
    p=(p_numerator/p_denominator)*(truth_right/total_pos)

    return p.sum()

class AverageMeter():
    def __init__(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, ncount=1):
        self.val=val
        self.sum+=val*ncount
        self.count+=ncount
        self.avg=self.sum/self.count


if __name__=="__main__":
    a=torch.range(0,10,dtype=torch.float)
    print(a)
