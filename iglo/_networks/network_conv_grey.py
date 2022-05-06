from IPython.core.debugger import set_trace

import torch
import torch.nn as nn

#layer_factor=16 if args.dim==2 else 128        
#Q = Q_MNIST_BN(layer_factor = 16, z_dim = 10)
#G = G_MNIST_BN(layer_factor = 16, z_dim =10)

# for energy,    
# if basic energy is not enough, consider g_depth

class G_MNIST_BN(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 10, nc=1,w=28, device = "cuda",**kw):
        super(G_MNIST_BN, self).__init__()
        # Define Layers
        preprocess = nn.Sequential(
            nn.Linear(z_dim, 4*4*4*layer_factor),
            nn.BatchNorm1d(4*4*4*layer_factor),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*layer_factor, 2*layer_factor, 5),
            nn.BatchNorm2d(2*layer_factor),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*layer_factor, layer_factor, 5),
            nn.BatchNorm2d(layer_factor),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(layer_factor, nc, 8, stride=2)
        # Define Network Layers
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        #self.sigmoid = nn.Sigmoid()
        self.d = layer_factor
        self.nc = nc
        self.w = w
        self.to(device)
        self.device =device
        self.train()
        
    # Define forward function
    def forward(self, z):
        #print("G!")
        #z = atanh(z)
        output = self.preprocess(z)

        output = output.view(-1, 4*self.d, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
    
        #output = self.sigmoid(output)
        return output.view(-1, self.nc, self.w, self.w)  #.view(-1, self.w*self.w)#

class Q_MNIST_mx(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 8, nc = 1, w=28, device = "cuda",**kw):
        
        super(Q_MNIST_mx, self).__init__()
        self.net1 = Q_MNIST_BN(layer_factor,z_dim, nc, w, device)
        self.net2 = Q_MNIST_BN(layer_factor,z_dim, nc, w, device)
        self.net3 = Q_MNIST_BN(layer_factor,z_dim, nc, w, device)
        self.comb = Q_MNIST_BN(layer_factor,3, nc, w, device)
        self.bias = Q_MNIST_BN(layer_factor,z_dim, nc, w, device)
        
        self.linear = nn.Linear(z_dim,z_dim)
        
        self.nc = nc
        self.w = w
        self.D = 4*4*4*layer_factor
        
    def forward(self, input):
        n = input.shape[0]
        input = input.view(-1, self.nc, self.w, self.w)
        weight = self.comb(input)
        bias = self.bias(input)
        #set_trace()
        out = torch.stack([self.net1(input),self.net2(input),self.net2(input)], dim=2)*weight.view(-1, 1,3) 
        out = out.sum(dim=2)+bias
        return out
'''
class Q_MNIST_BN(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 8, nc = 1, w=28, device = "cuda",**kw):
        print("just temporarily use the param umap's model")
        layer_factor=32 #param_UMAP
        
        
        super(Q_MNIST_BN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, layer_factor, 3, stride=(2,2)),
            nn.ReLU(True),
            nn.Conv2d(layer_factor, 2*layer_factor, 3, stride=(2,2)),
            nn.ReLU(True)
        )
        
        self.linears = nn.Sequential(
            nn.Linear(64*6*6, 256),
            nn.ReLU(True),
            nn.Linear(256,256),
            nn.ReLU(True),
            nn.Linear(256,2)
        )

        self.to(device)
        self.device =device
        self.nc = nc
        self.w = w
    def forward(self, input):
        #print("Q!")
        #set_trace()
        n = input.shape[0]
        input = input.view(-1, self.nc, self.w, self.w)
        out = self.main(input)

        out = out.view(n, -1)
        out = self.linears(out)
        return out

'''    
    
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

class Q_coil_BN(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 8, nc = 1, w=28, device = "cuda",**kw):
        super(Q_coil_BN, self).__init__()
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
        if layer_factor ==16:
            D = 4*4*4*16 *16
        elif layer_factor ==32:
            D = 4*4*4*16 * 32
        elif layer_factor ==64: 
            D = 4*4*4*16 * 32 *2
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
    
class two_input_critic_MNIST_BN(Q_MNIST_BN): 
    
    def __init__(self, device="cuda", nc = 1, w=28, layer_factor = 64):
        super().__init__(z_dim=1,layer_factor = layer_factor,nc=nc, w=w,  device = device) 
        self.output = nn.Linear(112*layer_factor, 1)
        
    def merger(self, x, y): 
        return torch.cat([x,y], dim = 1).reshape(-1, 1,  2*self.w, self.w)
        
    def forward(self, x, y):
        #TODO: how to mergy two img vectors
        data = self.merger(x,y)
        out = self.main(data)
        out = out.view(-1, 112*self.d)
        out = self.output(out)
        return out


class Q_MNIST(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 8, nc = 1, w=28, device = "cuda",**kw):
        super(Q_MNIST, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(nc, layer_factor, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(layer_factor, 2*layer_factor, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2*layer_factor, 4*layer_factor, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        #self.tanh = nn.Tanh()
        self.main = main
        self.output = nn.Linear(4*4*4*layer_factor, z_dim)
        self.d = layer_factor
        self.nc = nc
        self.w = w
        self.to(device)
        self.device =device
    def forward(self, input):
        #print("Q!")
        input = input.view(-1, self.nc, self.w, self.w)
        out = self.main(input)
        out = out.view(-1, 4*4*4*self.d)
        out = self.output(out)
        #out = self.tanh(out)
        return out

class two_input_critic_MNIST(Q_MNIST): 
    
    def __init__(self, device="cuda", nc = 1, w=28, layer_factor = 64):
        super().__init__(z_dim=1,layer_factor = layer_factor,nc=nc, w=w,  device = device) 
        self.output = nn.Linear(112*layer_factor, 1)
        #those dimension settings are important
        
    def merger(self, x, y): 
        return torch.cat([x,y], dim = 1).reshape(-1, 1,  2*self.w, self.w)
        
    def forward(self, x, y):
        #TODO: how to mergy two img vectors
        data = self.merger(x,y)
        out = self.main(data)
        out = out.view(-1, 112*self.d)
        out = self.output(out)
        return out
       
class G_MNIST(nn.Module):
    def __init__(self,layer_factor = 64,z_dim = 10, nc=1,w=28, device = "cuda",**kw):
        super(G_MNIST, self).__init__()
        # Define Layers
        preprocess = nn.Sequential(
            nn.Linear(z_dim, 4*4*4*layer_factor),
            nn.BatchNorm1d(4*4*4*layer_factor),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*layer_factor, 2*layer_factor, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*layer_factor, layer_factor, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(layer_factor, nc, 8, stride=2)
        # Define Network Layers
        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        #self.sigmoid = nn.Sigmoid()
        self.d = layer_factor
        self.nc = nc
        self.w = w
        self.to(device)
        self.device =device
        self.train()
        
    # Define forward function
    def forward(self, z):
        #print("G!")
        #z = atanh(z)
        output = self.preprocess(z)

        output = output.view(-1, 4*self.d, 4, 4)
        output = self.block1(output)
        output = output[:, :, :7, :7]
        output = self.block2(output)
        output = self.deconv_out(output)
    
        #output = self.sigmoid(output)
        return output.view(-1, self.w*self.w)#.view(-1, self.nc, self.w, self.w) 

