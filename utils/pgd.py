import torch
import torch.nn as nn


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
    X_random = 2 * (torch.rand_like(X) - 0.5) * epsilon
    X_pgd = torch.clamp(X.detach() + X_random, clip_min, clip_max)
    X_pgd.requires_grad = True

    for i in range(num_steps):
        pred = model(X_pgd)
        loss = criterion(pred, y)

        if print_process:
            print(f'iteration {i}, loss:{loss.item():.4f}')

        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        if euclidean:
            eta = (X_pgd.data - X.data).renorm(p=2, dim=0, maxnorm=epsilon)
        else:
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd
