#keras has some version issues. So, I will save mnist, fmnist from torch, and will read it manually later

from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np

#MNIST
ds = torchvision.datasets.MNIST('/home/kim2712/Desktop/data',
                                train=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor()]))
ds_tst = torchvision.datasets.MNIST('/home/kim2712/Desktop/data',
                                    train=False,
                                    transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor()]))
loader = DataLoader(ds, batch_size=60000, shuffle=False)
loader_tst = DataLoader(ds, batch_size=10000, shuffle=False)

for (x_train, y_train) in loader: break
for (x_test, y_test) in loader_tst: break

path = "/home/kim2712/Desktop/data/MNIST/JK_np"
for (name,array) in [("x_train",x_train),("y_train",y_train), ("x_test",x_test), ("y_test",y_test)]:
    with open(path+"/"+name+'.npy', 'wb') as f:
        np.save(f, array)

#F-MNIST
ds = torchvision.datasets.FashionMNIST('/home/kim2712/Desktop/data',
                                       train=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()]))
ds_tst = torchvision.datasets.FashionMNIST('/home/kim2712/Desktop/data',
                                           train=False,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor()]))
loader = DataLoader(ds, batch_size=60000, shuffle=False)
loader_tst = DataLoader(ds, batch_size=10000, shuffle=False)

for (x_train, y_train) in loader: break
for (x_test, y_test) in loader_tst: break

path = "/home/kim2712/Desktop/data/FashionMNIST/JK_np"
for (name,array) in [("x_train",x_train),("y_train",y_train), ("x_test",x_test), ("y_test",y_test)]:
    with open(path+"/"+name+'.npy', 'wb') as f:
        np.save(f, array)