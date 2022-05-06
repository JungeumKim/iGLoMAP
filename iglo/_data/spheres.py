#from : https://github.com/hyungkwonko/umato/blob/master/data/spheres/sphere_generation.py
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import rainbow

def dsphere(n=100, d=2, r=1, noise=None):

    data = np.random.randn(n, d)

    # Normalization
    data = r * data / np.sqrt(np.sum(data ** 2, 1)[:, None])

    if noise:
        data += noise * np.random.randn(*data.shape)

    return data


def create_sphere_dataset(total_samples=10000, d=100, n_spheres=10, r=5, r_out=25, var=1.0, seed=42, plot=False, frac = 2):
    
    

    variance = r / np.sqrt(d-1) * var

    np.random.seed(42)
    shift_matrix = np.random.normal(0, variance, [n_spheres, d])
    
    np.random.seed(seed)
    spheres = []
    n_datapoints = 0
    n_samples = total_samples // (frac * n_spheres)

    for i in np.arange(n_spheres):
        sphere = dsphere(n=n_samples, d=d, r=r)
        sphere_shifted = sphere + shift_matrix[i, :]
        spheres.append(sphere_shifted)
        n_datapoints += n_samples

    # Big surrounding sphere:
    n_samples_big = total_samples - n_datapoints
    big = dsphere(n=n_samples_big, d=d, r=r_out)
    spheres.append(big)
    n_datapoints += n_samples_big

    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colors = rainbow(np.linspace(0, 1, n_spheres+1))
        for data, color in zip(spheres, colors):
            ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=[color], s=5.0)
        plt.savefig("sample.png")

    dataset = np.concatenate(spheres, axis=0)

    labels = np.zeros(n_datapoints)
    label_index = 0
    for index, data in enumerate(spheres):
        n_sphere_samples = data.shape[0]
        labels[label_index : label_index + n_sphere_samples] = index
        label_index += n_sphere_samples

    return dataset, labels

def spheres_with_circle(N=2000, N_line=300, i=0, j=1,seed=1):
    r = 25
    X, Y = create_sphere_dataset(total_samples=N, d=101, n_spheres=10, r=5, r_out=25, var=1.0, seed=seed)
    circle = np.random.randn(N_line, 2)
    circle = r * circle / np.sqrt(np.sum(circle ** 2, 1)[:, None])
    data = np.zeros((101, N_line))

    data[i] = circle[:, 0]
    data[j] = circle[:, 1]

    line_c = np.ones(N_line) * 12

    X = np.vstack([X, data.transpose()])
    Y = np.concatenate([Y, line_c])
    return X, Y