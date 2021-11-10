import os
import torch
import config as cfg
import torch.distributed as dist

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

def compute_denominator(y_label, mean):
    return (y_label-mean)**2

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