from __future__ import print_function
import argparse
import numpy as np
from encodingGAN._data.data_giver import get_dataset
import os
import pathlib
from IPython.core.debugger import set_trace
import torch
from encodingGAN.evals.evaluator import MeasureCalculator
#worktree: enc_44d9b36

def parse_args():
    #Arguments:
    parser = argparse.ArgumentParser(description='Arguments.')

    #new arguments:
    parser.add_argument('--method', default ="umap", help="[umap, pacmap, isomap, tsne,phate]")
    parser.add_argument('--dataset', default ="spheres", help = '["spheres","sparse_egg12", "mammoth","mnist_np", "coil20","hier"]')
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--PP', type=int, default=15)
    parser.add_argument('--z_dir', default="/home/kim2712/Desktop/research/encodingGAN/_data_cache/baselines")
    parser.add_argument('--n_neighbor', default=None)

    args = parser.parse_args()    
    return args

def path_giver(args):
    path = os.path.join(args.z_dir, args.dataset, str(args.N), args.method)
    if args.n_neighbor is not None:
        path = path +"_nb_" + str(args.n_neighbor)

    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path

def main(args):

    data, y = get_dataset(args.dataset, n= args.N)
    if args.n_neighbor is None:
        kw={}
    else:
        if args.method == "tsne": 
            kw={"perplexity": args.n_neighbor}
        elif args.method == "phate":
            kw={"knn": args.n_neighbor}
        else:
            kw={"n_neighbors": args.n_neighbor}

    if args.method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, **kw)
    elif args.method == "tsne":
        from sklearn.manifold import TSNE
        reducer = tsne = TSNE(n_components = 2,**kw)
    elif args.method == "phate":
        import phate
        reducer = phate.PHATE(n_components=2, **kw)
    elif args.method == "pacmap":
        import sys
        sys.path.append("/home/kim2712/Desktop/research/Othermethods/PaCMAP/source_code")
        import pacmap.pacmap as pacmap
        reducer = pacmap.PaCMAP(n_dims=2, **kw)
    elif args.method == "isomap":
        from sklearn.manifold import Isomap
        reducer = Isomap(n_components=2, **kw)
    else:
        assert False, "Give me a valid method name please."
    print(F"reducer generated from {args.method}")
    print(F"embedding of {args.dataset} optimization begin")

    z = reducer.fit_transform(data)
    x = data
    
    print("embedding done")
    path = path_giver(args)

    with open(path+"/z.npy", "wb") as file:
        np.save(file, z, allow_pickle=True)
    with open(path+"/x.npy", "wb") as file:
        np.save(file, x, allow_pickle=True)
    with open(path+"/y.npy", "wb") as file:
        np.save(file, y, allow_pickle=True)
    print(F"embedding of {args.method} saved at {path}")
    
    
    if (args.dataset in ["mnist", "kmnist", "fmnist", "cifar", "coil20"]) or ("np" in args.dataset) or ("pca" in args.dataset):
        local=True
        knn_n_neighbors=[1, 3, 5, 10, 15, 20, 25, 30]
        
    elif ("hier" in args.dataset) or (args.dataset == "spheres"):
        local=True
        knn_n_neighbors=[10, 30, 50, 100, 150, 200, 250, 300]
        
    elif (args.dataset in ["scurve", "severe", "sparse_egg12"]):
        local=False
        knn_n_neighbors=None
    
    for dist in ["L2", "global", "rescaled_global"]: 
        calc = MeasureCalculator(k_max=200, X=x, Z=z,Y=y, distance=dist, local=local, knn_n_neighbors=knn_n_neighbors, dataset = args.dataset,
                                PP = args.PP)
        result = calc.run(single=False)
        torch.save(result, os.path.join(path, "eval_" + dist))
        print("saved at"+ os.path.join(path,  "eval_" + dist))
        

            
if __name__ == '__main__':
    args = parse_args()

    main(args)
    exit()


