import utils
import torch
import config as cfg

class FocalLoss(torch.nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.gamma=cfg.focal_loss_gamma
        self.alpha=cfg.focal_loss_alpha

    def forward(self, logits, targets):
        num_classes = logits.shape[1]
        device = targets.device
        dtype = targets.dtype
        class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)

        t = targets.unsqueeze(1)
        p = logits

        term1 = (1 - p) ** self.gamma * torch.log(p)
        term2 = p ** self.gamma * torch.log(1 - p)

        loss = -(t == class_range).float() * term1 * self.alpha - ((t != class_range) * (t >= 0)).float() * term2 * (
                    1 - self.alpha)

        return loss.sum()

class ClassLoss(torch.nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()
        self.loss=FocalLoss()

    def forward(self, predict, y):
        label = utils.encode(y)
        loss=self.loss(predict, label)/cfg.samples_per_gpu

        return loss
