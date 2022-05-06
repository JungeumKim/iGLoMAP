from IPython.core.debugger import set_trace

import torch
import torch.nn as nn

# The 2d model of DOT is simply changed into torch modelssame model 
# But, the latent dim is reduced to 1.

### for 2d
class G_2dim(torch.nn.Module):
    def __init__(self, device="cuda",dim=2,  z_dim=1, ch_fact = 512, leaky=0.1):
        super().__init__()
        self.dim=dim
        self.z_dim = z_dim
        self.non_linear = torch.nn.LeakyReLU(leaky) if leaky>0 else torch.nn.ReLU()
        
        
        # TODO: init - it may affect the results.

        self.l0 = torch.nn.Linear(z_dim, ch_fact)
        self.l1 = torch.nn.Linear(ch_fact, ch_fact)
        self.l2 = torch.nn.Linear(ch_fact, ch_fact)
        self.l3 = torch.nn.Linear(ch_fact, ch_fact)
        self.l4 = torch.nn.Linear(ch_fact, dim)
        
        self.to(device)
        self.device = device
        
    def forward(self, z):
        h = self.non_linear(self.l0(z))
        h = self.non_linear(self.l1(h))
        h = self.non_linear(self.l2(h))
        h = self.non_linear(self.l3(h))
        h = self.l4(h)
        
        return h.view(len(z), self.dim)

    def sampling(self, batchsize):
        z = torch.normal(size=(batchsize, self.z_dim), device = self.device)
        return self(z)

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

class two_input_critic_2dim(Q_2dim): #TODO: if it does not work, then I need to increase the model capacity larger then Q_2dim
    def __init__(self, device="cuda", dim=2, leaky=0.1):
        super().__init__(dim = dim*2, z_dim=1, leaky=leaky, device = device) #those dimension settings are important
    
    def merger(self, x, y): 
        return torch.cat([x,y], dim = 1) #n should be the same and dim is increasd by 2 since merging.
        
    def forward(self, x, y):
        data = self.merger(x,y)
        return super().forward(data)
   
    

class basic_energy(torch.nn.Module): 
    def __init__(self,input_dim=2, out_dim=1, ch_factor = 64, tanh=False, device = "cpu"):
        super().__init__()
        
        self.input_dim = input_dim
        E = torch.nn.Sequential()
        E.add_module("L1",nn.Linear(input_dim, ch_factor*input_dim))
        E.add_module("relu1",nn.LeakyReLU(0.2))
        E.add_module("L2",nn.Linear(ch_factor*input_dim, ch_factor*input_dim))
        E.add_module("relu2",nn.LeakyReLU(0.2))

        E.add_module("L3",nn.Linear(ch_factor*input_dim, ch_factor*input_dim))
        E.add_module("relu3",nn.LeakyReLU(0.2))

        E.add_module("L4",nn.Linear(ch_factor*input_dim, out_dim))
        
        if tanh: 
            E.add_module("Tanh",nn.Tanh())
            
        self.E = E
        self.to(device)
        self.device = device
        
    def forward(self,z):
        #print("E!")
        assert z.shape[1]==self.input_dim, F"The input must be n x {self.input_dim} tensor"
        return self.E(z)

class basic_energy_comp(basic_energy): 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def forward(self,z):
        #set_trace()
        assert z.shape[1]==self.input_dim, F"The input must be n x {self.input_dim} tensor"
        return self.E(z) + (0.5 * z.norm(dim=1)**2).view(-1, 1)


class sequential_energy(torch.nn.Module):
    def __init__(self, device="cuda", input_dim=1, T=6):#, gauss_tilted=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.non_linear = torch.nn.LeakyReLU(0.1)
        self.T = T
        
        # TODO: init - it may affect the results.
        energy_seq=[basic_energy(input_dim, 1, (t+1)*10) for t in reversed(range(T))]
        
        self.energy_seq = torch.nn.ModuleList(energy_seq)
        
        self.to(device)
        self.device = device
        #self.gauss_tilted = gauss_tilted
        
    def forward(self, x=1, t=1):
        
        if isinstance(t, int):
            t = torch.ones(len(x), device = self.device) * t
            
        E = torch.zeros((len(x),1), device=self.device)
            
        for i, energy in enumerate(self.energy_seq):
            E  += energy(x) * (t == i).view(-1,1).float()
        
        #if self.gauss_tilted: 
        #    gauss_energy = (0.5 * x.norm(dim=1)**2).view(-1,1)
        #    return E + gauss_energy
        #else:
        #    return E
        return E


class Q_2dim_deep(torch.nn.Module):
    def __init__(self, device="cuda", dim=2, z_dim=1, leaky=0.1, factor = 512):
        super().__init__()
        self.dim =dim
        self.non_linear = torch.nn.LeakyReLU(leaky) if leaky>0 else torch.nn.ReLU()
        # TODO: init - it may affect the results.
        
        self.l1 = torch.nn.Linear(dim, factor)
        self.l2 = torch.nn.Linear(factor, factor)
        self.l21 = torch.nn.Linear(factor, factor)
        self.l22 = torch.nn.Linear(factor, factor)
        self.l23 = torch.nn.Linear(factor, factor)
        self.l3 = torch.nn.Linear(factor, factor)
        self.l4 = torch.nn.Linear(factor, z_dim)
        
        self.to(device)
        self.device = device
        
    def forward(self, x):
        x = x.view(-1, self.dim)
        h = self.non_linear(self.l1(x))
        h = self.non_linear(self.l2(h))
        h = self.non_linear(self.l21(h))
        h = self.non_linear(self.l22(h))
        h = self.non_linear(self.l23(h))        
        h = self.non_linear(self.l3(h))
        h = self.l4(h)
        return h
    
class Q_2dim_deeper(torch.nn.Module):
    def __init__(self, device="cuda", dim=2, z_dim=1, leaky=0.1, factor = 512):
        super().__init__()
        self.dim =dim
        self.non_linear = torch.nn.LeakyReLU(leaky) if leaky>0 else torch.nn.ReLU()
        # TODO: init - it may affect the results.
        
        self.l1 = torch.nn.Linear(dim, factor)
        self.l2 = torch.nn.Linear(factor, factor)
        self.l21 = torch.nn.Linear(factor, factor)
        self.l22 = torch.nn.Linear(factor, factor)
        self.l23 = torch.nn.Linear(factor, factor)
        self.l24 = torch.nn.Linear(factor, factor)
        self.l25 = torch.nn.Linear(factor, factor)
        self.l26 = torch.nn.Linear(factor, factor)
        self.l3 = torch.nn.Linear(factor, factor)
        self.l4 = torch.nn.Linear(factor, z_dim)
        
        self.to(device)
        self.device = device
        
    def forward(self, x):
        x = x.view(-1, self.dim)
        h = self.non_linear(self.l1(x))
        h = self.non_linear(self.l2(h))
        h = self.non_linear(self.l21(h))
        h = self.non_linear(self.l22(h))
        h = self.non_linear(self.l23(h))        
        h = self.non_linear(self.l24(h))        
        h = self.non_linear(self.l25(h))        
        h = self.non_linear(self.l26(h))        
        h = self.non_linear(self.l3(h))
        h = self.l4(h)
        return h