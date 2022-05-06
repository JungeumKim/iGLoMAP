import torch
from torch.utils.data import Dataset, DataLoader

class Np_dataset(Dataset): 
    def __init__(self, X_np, Y_np=None, return_idx = False,float_data=True):

        self.X = torch.from_numpy(X_np)
        if float_data:
            self.X = self.X.float()
        self.len = self.X.shape[0]
        self.return_idx = return_idx
        
        if Y_np is not None:
            self.Y = torch.from_numpy(Y_np)
            self.y = True
        else:
            self.y = False
            
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if self.return_idx: 
            if self.y:
                return self.X[idx], self.Y[idx], idx
            else:
                return self.X[idx], idx
        else:
            if self.y:
                return self.X[idx], self.Y[idx]
            else:
                return self.X[idx]

