import torch
import torch.nn as nn

from dataclasses import mean, std

def vector_norm(x):
    return x.view(x.shape[0], -1).norm(dim=1).mean()

def pgd_attack(model,
               X,
               y,
               epsilon=0.03,
               clip_max=1.0,
               clip_min=0.0,
               num_steps=3,
               step_size=0.01,
               euclidean=True,
               criterion=nn.CrossEntropyLoss(),
               print_process=False,
               ):
    X.mul_(std).add_(mean)
    if euclidean:
        l = len(X.shape) - 1
        rp = torch.randn_like(X)
        rp_norm = rp.view(rp.shape[0], -1).norm(dim=1).view(-1, *([1]*l))
        X_random = epsilon * rp / (rp_norm + 1e-10)
    else:
        X_random = 2 * (torch.rand_like(X) - 0.5) * epsilon
    X_pgd = torch.clamp(X.detach() + X_random, clip_min, clip_max)
    X_pgd.requires_grad = True

    for i in range(num_steps):
        pred = model(X_pgd)
        loss = criterion(pred, y)

        if print_process:
            print(f'iteration {i}, loss:{loss.item():.4f}')

        loss.backward()

        if euclidean:
            g = X_pgd.grad.data
            l = len(X.shape) - 1
            g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1]*l))
            eta = step_size * g.data / (g_norm + 1e-10)
        else:
            eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        if euclidean:
            eta = (X_pgd.data - X.data).renorm(p=2, dim=0, maxnorm=epsilon)
        else:
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = torch.clamp(X.data + eta, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    X_pgd.requires_grad = False
    X_pgd.sub_(mean).div_(std)
    return X_pgd
