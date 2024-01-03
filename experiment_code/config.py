import numpy as np
import seaborn as sns
from importlib import reload
import matplotlib.pyplot as plt
import torch
from IPython.core.debugger import set_trace
from os.path import join
import pathlib
import sys
import argparse

def parse_one():
    #Arguments:
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--exp_dir', default ="../configs/nat0.json")
    exp_dir, lest = parser.parse_known_args()
    return exp_dir, lest


def main(args):
    path = join(args.exp_dir,"result_data",F"n_data{args.n_data}", F"epoch{args.n_epoch}",
            F"tau_percentile_init{args.initial_tau_percentile}end{args.end_tau_percentile}",
            F"tau_init{args.initial_sigma}end{args.end_sigma}")
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    for data in ["hier","spheres","scurve","eggs","severe","mnist"]:
        X,Y = get_dataset("scurve",args.n_data)
        reducer = glomap.iGLoMAP(
                 show=False,
                 EPOCHS = args.n_epoch,
                 exact_mu = True,
                 use_mapper=False,
                 save_vis=True,
                 initial_sigma=args.initial_sigma,
                 end_sigma=args.end_sigma,
                 initial_tau_percentile = args.initial_tau_percentile,
                 end_tau_percentile = args.end_tau_percentile)
        p = reducer.fit_transform(X,Y)
        torch.save(reducer.Z_list, join(path,f"{data}_zlist.dat"))# "./result_data/glomap_severed_sig10.dat")


if __name__ == '__main__':

    #print(torch.__version__) #does the slurm error come from the version compatibility?

    args_one, args_lest = parse_one()
    sys.path.append(args_one.exp_dir)

    from config import parse_args

    args = parse_args(args_lest, name_space=args_one)

    if hasattr(args, "worktree"):
        sys.path.insert(0, args.worktree) #+ "/sourceCode")
    else:
        sys.path.insert(0, "/home/kim2712/Desktop/research/iglo")
    from iglo._data import get_dataset
    from iglo.manifold import glomap
    from iglo.evals import vis_tool
    from iglo._data.hierarch import hierarch_data

    main(args)
