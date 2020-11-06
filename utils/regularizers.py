import torch
import torch.nn as nn


class LinDLReg(nn.Module):
    # Implementation of https://arxiv.org/pdf/2011.00368.pdf
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, x, target=None):
        with torch.no_grad():
            x = x.view(x.size(0), -1)
            x_ext = torch.ones((x.size(0), x.size(1) + 1)).to(x.device)
            x_ext[:, : x.size(1)] = x
            x_ext_t = torch.transpose(x_ext, 0, 1)
            target = target.view(target.size(0), -1)
            try:
                z = torch.matmul(torch.matmul(x_ext_t, torch.inverse(torch.matmul(x_ext, x_ext_t))),
                                 target)
            except:
                print('LenDLReg: X is not a full rank matrix')
                return 0.

        return self.gamma * torch.norm(torch.matmul(x_ext, z) - target)