import torch
import config as cfg

def make_optimizer(model):
    if(hasattr(model,"module")):
        model=model.module

    optimizer=torch.optim.RMSprop(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay,momentum=cfg.momentum)
    return optimizer