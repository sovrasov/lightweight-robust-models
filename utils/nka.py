import torch
import torch.nn as nn

from .dataloaders import MEAN, STD
from .pgd import vector_norm

mean = torch.tensor(MEAN).cuda().view(1,3,1,1) / 255.
std = torch.tensor(STD).cuda().view(1,3,1,1) / 255.

def nka_attack(model,
               X,
               y,
               epsilon=0.1,
               clip_max=1.0,
               clip_min=0.0,
               num_steps=3,
               step_size=0.01,
               euclidean=True,
               criterion=nn.CrossEntropyLoss(),
               print_process=False,
               ):
    X.mul_(std).add_(mean)
    X_pgd = torch.clamp(X.detach(), clip_min, clip_max)
    X_pgd.requires_grad = True

    for i in range(num_steps):
        model.zero_grad()
        pred = model(X_pgd)
        loss = criterion(pred, y)

        if print_process:
            print(f'iteration {i}, loss:{loss.item():.4f}')

        loss.backward()

        X_pgd = X_pgd - epsilon * X_pgd.grad.data.sign()
        X_pgd = torch.clamp(X_pgd.data, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    X_pgd.requires_grad = False
    #print('\n')
    #print(torch.norm(X_pgd - X, p=float('inf')).item())
    #print('\n')
    X_pgd.sub_(mean).div_(std)
    return X_pgd


def adversarial_nka_loss(x, target):
    adv_conf_pred = nn.functional.softmax(x, dim=1)
    mean_conf = torch.max(adv_conf_pred)
    ent = torch.sum(- adv_conf_pred * torch.log(adv_conf_pred.clamp(min=10**-8)), dim=1).mean()
    #print('\n')
    #print(mean_conf.item())
    #print('\n')
    return - ent

def adv_rej_acc(x):
    adv_conf_pred = nn.functional.softmax(x, dim=1)
    return torch.sum(adv_conf_pred < 0.15) / torch.numel(adv_conf_pred)
