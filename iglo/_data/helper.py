import torch

def get_Z(net, loader, device="cpu"):
    Zs = []
    ys = []
    with torch.no_grad():
        for x, y in loader:
            Zs.append(net.scaled_embd(x.to(device)))
            ys.append(y)

        Z = torch.cat(Zs).cpu()
        y = torch.cat(ys).cpu()
    return Z.numpy(), y.numpy()


def get_X(loader, pretty=True):
    Xs = []
    with torch.no_grad():
        for x, y in loader:
            Xs.append(x)

        X = torch.cat(Xs)

    if pretty:
        n = X.shape[0]
        X = X.view(n,-1) #2d form
    return X.numpy()

