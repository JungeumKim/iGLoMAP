from IPython.core.debugger import set_trace
from iglo._data.spheres import create_sphere_dataset, spheres_with_circle
from iglo._data.hierarch import hierarch_data, hierarch_data2
from iglo._data.eggs import *
from iglo.Datasets import Np_dataset, Idx_dataset
import sklearn.datasets
import numpy as np
from sklearn.utils import check_random_state
import torch 
from os.path import join,exists

from torch.utils.data import DataLoader, Dataset

#shortest path:
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from iglo.manifold._util_manifold import compute_rescaled_dist
import pathlib

def get_precalc_graph(dataset, N_train, max_=True, n_neighbors = 5, return_predecessors=False,
                      directory="/home/kim2712/Desktop/data/shortest_distances"):
    path =  join(directory, dataset)
    path_D = join(path, str(N_train) + F"geo{n_neighbors}.npy")
    path_press = join(path, str(N_train) + F"press{n_neighbors}.npy")
    path_y = join(path, str(N_train) + F"Y{n_neighbors}.npy")
    path_x = join(path, str(N_train) + F"X{n_neighbors}.npy")
    if exists(path_D):
        D= np.load(path_D)
        print(F"The pre_calculated shortest distance matrix is loaded from {path}")
        X = np.load(path_x)
        Y = np.load(path_y)
        if return_predecessors:
            Press = np.load(path_press)
            return D,Press,X,Y
        return D, X,Y
    else:
        return None, None,None

def shortest_path_savior(dataset, N_train, max_=True, n_neighbors = 5,
                         directory="/home/kim2712/Desktop/data/shortest_distances"):
    # Solving path
    path = join(directory, dataset)
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    path_D = join(path, str(N_train) + F"geo{n_neighbors}.npy")
    path_press = join(path, str(N_train) + F"press{n_neighbors}.npy")
    path_y = join(path, str(N_train) + F"Y{n_neighbors}.npy")
    path_x = join(path, str(N_train) + F"X{n_neighbors}.npy")

    # Data handling
    X,Y = get_dataset(dataset, N_train,ignore_y =False)
    #set_trace()
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    knn_dists, knn_indices = nbrs.kneighbors(X)  # the first column is with the point itself.
    sigmas = knn_dists.sum(axis=1) / (n_neighbors - 1)
    rescaled_knn_dists_mat = compute_rescaled_dist(knn_indices, knn_dists, sigmas, 1, True)
    shortest_D, Press = shortest_path(rescaled_knn_dists_mat, directed=False,
                               return_predecessors=True)
    #Saving
    shortest_D = shortest_D.astype(np.float16)
    np.save(path_D, shortest_D)
    np.save(path_x, X)
    np.save(path_y, Y)
    np.save(path_press, Press)
    #DONE.
    print("shortest path saved")
    return shortest_D,Press, X, Y

def get_dataset(dataset, n, train=True, ignore_y =False, n_pca=6000, extra_col=False): 
    '''
        gives: scurve, spheres, severe. 
        return: X, Y (color or label)
        For mnist, if a part is given, I set the first n examples are given. 
            This is to match the pre-calculated distance of tsne and isomap.
    '''
    seed = 1 if train else 2

    # dataset"_np" or "_pca" case.
    if ("mnist" in dataset) :
        #I have "fmnist_np" and "fmnist_pca", "mnist_np" and "mnist_pca": both will be in np.ndarray.
        path = "/home/kim2712/Desktop/data/MNIST/JK_np"

        type_ = "train" if train else "test"
        X, Y = np.load(path + F"/x_{type_}.npy"), np.load(path + F"/y_{type_}.npy")
        # I vectorize data to make the data compatible for other visualizers.
        sz = X.shape[0]
        X = X.reshape(sz, -1)
        
        if (train and (n < 60000)) or (~train and n<10000):
            #random_select = np.random.permutation(np.arange(X.shape[0]))[:n]
            #X,Y = X[random_select], Y[random_select]
            X,Y = X[:n], Y[:n]
        if "pca" in dataset:
            # For both training and testset, we calculated pc vectors for the first n_pca training dataset.
            # Then, project the training or test set onto the pc subspace
            # I did not implement for general latent dimensions BUT ONLY 50 dimensions which I need.
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            x_train = np.load(path + F"/x_train.npy")
            pca.fit((x_train[:n_pca]/255).reshape(-1,28*28))
            X = pca.transform(X)

        if ignore_y: return X
        return X, Y


    elif dataset == "scurve":
        X, Y = sklearn.datasets.make_s_curve(n, random_state=seed)
        X = X.astype("float32")


    elif dataset == "spheres":
        seed = 42 if train else 1 #somehow the code from UMATO has seed 42. I don't know why they used this number.
        X, Y = create_sphere_dataset(total_samples=n, n_spheres=10,
        d=101, r=5, r_out=25, var=1.0, seed=seed, plot=False)


    elif dataset == "sparse_egg12":
        n_unif, n_hsph = egg_split(_min = -4, _max = 4, n=n, r=1, eggs=12 ,square = False, alpha=0.3)
        X, Y = get_12_egg(n_unif=n_unif, n_egg=n_hsph, seed = seed)


    elif "hier" in dataset:
        X, c_macro, c_meso, c_micro = hierarch_data(n=n, seed = seed)
        
        if extra_col: 
            Y = np.stack([c_macro, c_meso, c_micro]).transpose()
        elif "meso" in dataset:
            Y = c_meso
        elif "micro" in dataset:
            Y = c_micro
        else: 
            Y = c_macro
            

    elif dataset == "severe":
        # Code from sklearn
        random_state = check_random_state(seed)
        p = random_state.rand(n) * (2 * np.pi - 0.55)
        t = random_state.rand(n) * np.pi
        # Sever the poles from the sphere.
        indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
        
        x, y, z = np.sin(t[indices]) * np.cos(p[indices]), \
            np.sin(t[indices]) * np.sin(p[indices]), \
            np.cos(t[indices])
        X = np.array([x, y, z]).T
        Y = p[indices]

    if "egg" in dataset:
        X = X.transpose()
    if X.shape[0]>n: 
        np.random.seed(111)
        random_select = np.random.permutation(np.arange(X.shape[0]))[:n]
        X,Y = X[random_select], Y[random_select]
    if ignore_y: return X
    return X,Y


def get_loaders(n=1000, train=True, dataset = "mnist", batch_size=100, shuffle=False, index_ds = False):

    ds = get_dataset(dataset, n, train=train, ignore_y =False)

    if isinstance(dataset, Dataset) or isinstance(ds, torch.utils.data.dataset.Subset) or hasattr(ds,"_is_protocol"):
        if index_ds:
            ds = Idx_dataset(ds)
    else:
        
        X, Y = ds
        if index_ds:
            ds = Np_dataset(X, return_idx=True)
        else:
            ds = Np_dataset(X, Y, return_idx = False)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return loader

    
