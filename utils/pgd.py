import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pgd_attack(model,
                  X,
                  y,
                  epsilon=0.03,
                  clip_max=1.0,
                  clip_min=0.0,
                  num_steps=3,
                  step_size=0.01,
                  euclidean=True,
                  print_process=False):
    #out = model(X)
    #err = (out.data.max(1)[1] != y.data).float().sum()
    #TODO: find a other way
    #device = X.device
    #imageArray = X.detach().cpu().numpy()
    #np.random.uniform(-epsilon, epsilon, X.shape)
    #imageArray = np.clip(imageArray + X_random, 0, 1.0)
    #X_pgd = torch.tensor(imageArray).to(device).float()
    X_random = 2 * (torch.rand_like(X) - 0.5) * epsilon
    X_pgd = torch.clamp(X.detach() + X_random, clip_min, clip_max)
    X_pgd.requires_grad = True
    criterion = nn.CrossEntropyLoss()

    for i in range(num_steps):
        pred = model(X_pgd)
        loss = criterion(pred, y)

        if print_process:
            print("iteration {:.0f}, loss:{:.4f}".format(i,loss))

        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        if euclidean:
            #eta = epsilon * F.normalize((X_pgd.data - X.data).view(X.size(0), -1)).view(*X.shape)
            eta = (X_pgd.data - X.data).renorm(p=2, dim=0, maxnorm=epsilon)
        else:
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()


    return X_pgd
