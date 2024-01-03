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

def parse_args(args, name_space):
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--worktree',
                        default = "/home/kim2712/Desktop/research/iglo",
                        help = "parent directory")
    parser.add_argument('--exp_dir', default = "./",  help= 'global path')

    #method
    #parser.add_argument('--dataset', default ="hier",  help=["hier, spheres"])

    #data:
    parser.add_argument('--n_data',  type=int, default = 6000)
    parser.add_argument('--n_epoch', type=int, default = 300,
                        help= 'latent dimension')
    #parser.add_argument('--ee',       type=float, default = 1)

    parser.add_argument('--initial_sigma', type=float, default = 10)
    parser.add_argument('--end_sigma',  type=float, default = 10)
    parser.add_argument('--initial_tau_percentile', type=float, default = 75)
    parser.add_argument('--end_tau_percentile',  type=float, default = 25)
    #learning setting:
    args = parser.parse_args(args, namespace=name_space) #parser.parse_args()
    return args
