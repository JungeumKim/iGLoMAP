import torch
from torch.utils.data import Dataset
from IPython.core.debugger import set_trace

''' Usage: 
Need: data X

get_r_nb_idx_gen_tsne: 
    This utilizes t-SNE implementation of sklearn to get the distances and P.
    The random neighbor generation will be based on this P or distances.

    Main hyper parameters: 

        symmetricP: if False, uses a conditional P (transition mat) 
                    if True,  uses a normalized (cond_P+cond_P.T)/2
    
        distance_only: if true, not use P but only distance for random selection among KNN neighbors
        
        desired_perplexity: used in two ways
                1. n_neighbors = min(n_samples - 1, int(3. * desired_perplexity + 1)) 
                   i.e.,n_nb = three times of perplexity
                2. perplexity for calculating P (tuning sigma of the tsne P model)

Usage: 
    
    from neighbor_dataset import Neighbor_dataset
    from manifold.itsne import random_nb, get_r_nb_idx_gen_tsne
    from torch.utils.data import DataLoader
    
    random_idx_generator = get_r_nb_idx_gen_tsne(X, desired_perplexity = 5,symmetricP = True, distance_only=False)
    ds = neighbor_dataset(X, Y, neighbor_rule = random_idx_generator, return_idx = True)
    t_loader = DataLoader(ds, batch_size=100, shuffle=True, num_workers=4)

    for x, y, idx, xn, yn, idxn in t_loader: 

'''
class Neighbor_dataset(Dataset): 
    
    def __init__(self, X_np, Y_np=None, neighbor_rule=None, return_idx = False):
        '''
        If Y_np is None, should be specify Y_np = None
        '''
        self.X = torch.from_numpy(X_np).float()
        
        if Y_np is None: 
            self.none_y = True
        else: 
            self.none_y = False
            self.Y = torch.from_numpy(Y_np)
            
        self.len = self.X.shape[0]
        self.return_idx = return_idx
        
        self.neighbor_rule = neighbor_rule
        
    def __len__(self):
        return self.len
    
    
    def __getitem__(self, idx):
        
        nb_idx, neighbors, mu_ij_of_neighbors  = self.neighbor_rule(idx)
        if self.none_y:
            if self.return_idx:
                # this is the case used in iglomap.
                return self.X[idx], idx, self.X[nb_idx], nb_idx,neighbors, mu_ij_of_neighbors
            else:
                return self.X[idx], self.X[nb_idx]
        else: 
            if self.return_idx: 
                return self.X[idx], self.Y[idx], idx, self.X[nb_idx], self.Y[nb_idx], nb_idx
            else:
                return self.X[idx], self.Y[idx], self.X[nb_idx], self.Y[nb_idx]


class Neighbor_dataset_f(Dataset):

    def __init__(self,dataset, neighbor_rule=None, return_idx=False,ignore_y = True):
        '''
        If Y_np is None, should be specify Y_np = None
        '''
        self.dataset= dataset
        self.return_idx = return_idx
        self.neighbor_rule = neighbor_rule
        self.ignore_y = ignore_y 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        nb_idx = self.neighbor_rule(idx)
        if self.ignore_y: 
            if self.return_idx:
                return self.dataset[idx][0], idx, self.dataset[nb_idx][0], nb_idx
            else:
                return self.dataset[idx][0], self.dataset[nb_idx][0]            
        else:
            if self.return_idx:
                return *self.dataset[idx], idx, *self.dataset[nb_idx], nb_idx
            else:
                return *self.dataset[idx], *self.dataset[nb_idx]
