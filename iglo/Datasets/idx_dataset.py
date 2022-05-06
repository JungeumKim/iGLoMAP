import torch
from torch.utils.data import Dataset

class Idx_dataset(Dataset): 
    def __init__(self, dataset, ignore_y=True):
        assert (isinstance(dataset, Dataset) or isinstance(dataset, torch.utils.data.dataset.Subset))
        self.dataset = dataset
        self.ignore_y = ignore_y
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.ignore_y:
            return self.dataset.__getitem__(idx)[0], idx
        else:
            return *self.dataset.__getitem__(idx), idx