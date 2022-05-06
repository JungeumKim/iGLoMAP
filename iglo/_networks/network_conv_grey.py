from IPython.core.debugger import set_trace

import torch.nn as nn

class Q_MNIST_BN(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 8, nc = 1, w=28, device = "cuda",**kw):
        super(Q_MNIST_BN, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(nc, layer_factor, 5, stride=2, padding=2),
            nn.BatchNorm2d(layer_factor),
            nn.ReLU(True),
            nn.Conv2d(layer_factor, 2*layer_factor, 5, stride=2, padding=2),
            nn.BatchNorm2d(2*layer_factor),
            nn.ReLU(True),
            nn.Conv2d(2*layer_factor, 4*layer_factor, 5, stride=2, padding=2),
            nn.BatchNorm2d(4*layer_factor),
            nn.ReLU(True),
        )
        #self.tanh = nn.Tanh()
        self.main = main
        D = 4*4*4*layer_factor
        
        self.D = D
        self.output = nn.Linear(D, z_dim)
        self.d = layer_factor
        self.nc = nc
        self.w = w
        self.to(device)
        self.device =device
    def forward(self, input):
        #print("Q!")
        #set_trace()
        n = input.shape[0]
        input = input.view(-1, self.nc, self.w, self.w)
        out = self.main(input)
        out = out.view(-1, self.D)
        #out = out.view(n, -1)
        out = self.output(out)
        #out = self.tanh(out)
        return out
