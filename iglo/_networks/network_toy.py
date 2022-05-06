from IPython.core.debugger import set_trace

import torch

### for 2d

class Q_2dim(torch.nn.Module):
    def __init__(self, device="cuda", dim=2, z_dim=1, leaky=0.1, factor = 512):
        super().__init__()
        self.dim =dim
        self.non_linear = torch.nn.LeakyReLU(leaky) if leaky>0 else torch.nn.ReLU()
        # TODO: init - it may affect the results.

        self.l1 = torch.nn.Linear(dim, factor)
        self.l2 = torch.nn.Linear(factor, factor)
        self.l3 = torch.nn.Linear(factor, factor)
        self.l4 = torch.nn.Linear(factor, z_dim)

        self.to(device)
        self.device = device

    def forward(self, x):

        h = self.non_linear(self.l1(x))
        h = self.non_linear(self.l2(h))
        h = self.non_linear(self.l3(h))
        h = self.l4(h)
        return h