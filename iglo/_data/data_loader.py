import numpy as np
import sklearn.datasets


#### data generators are derived from https://github.com/lukovnikov/improved_wgan_training/blob/master/gan_toy.py
# Copyright (c) 2017 Ishaan Gulrajani
# Released under the MIT license
# https://github.com/lukovnikov/improved_wgan_training/blob/master/LICENSE

def random_manifold_updim(data, input_dim=2, out_dim = 10):

    assert data.shape[-1] == input_dim
    
    random_linear_map = np.random.rand(input_dim,out_dim)
    
    return np.matmul(data, random_linear_map)


def prepare_swissroll_data(BATCH_SIZE=1000, dim=2):
    data = sklearn.datasets.make_swiss_roll(
                    n_samples=BATCH_SIZE,
                    noise=0.25
                )[0]
    data = data.astype('float32')[:, [0, 2]]
    data /= 7.5 # stdev plus a little
    
    if dim > 2: 
        data = random_manifold_updim(data, input_dim=2, out_dim = dim)
        
    return data.astype('float32')

def prepare_25gaussian_data(BATCH_SIZE=1000, dim=2):
    dataset = []
    for i in range(BATCH_SIZE//25):
        for x in range(-2, 3):
            for y in range(-2, 3):
                point = np.random.randn(2)*0.05
                point[0] += 2*x
                point[1] += 2*y
                dataset.append(point)
    dataset = np.array(dataset, dtype=np.float32)
    np.random.shuffle(dataset)
    dataset /= 2.828 # stdev

    if dim > 2: 
        dataset = random_manifold_updim(dataset, input_dim=2, out_dim = dim)
    
    return dataset.astype('float32')

def prepare_s_curve4(BATCH_SIZE=4000, k = 4, dim=3):
    
    X, color = sklearn.datasets.make_s_curve(int(BATCH_SIZE/4), random_state=0)

    X1 = X +np.array([k,0,k])
    X2 = X +np.array([-k,0,k])
    X3 = X +np.array([-k,0,-k])
    X4 = X +np.array([k,0,-k])
    
    X_list = [X1,X2,X3,X4]
    color_list = [color for i in range(4)]
    
    dataset = np.concatenate(X_list)
    color = np.concatenate(color_list)
    
    if dim > 3: 
        dataset = random_manifold_updim(dataset, input_dim=3, out_dim = dim)
        
    return dataset.astype('float32'), color


def prepare_spiral_data(BATCH_SIZE=1000, couple=True, unif_noise=0, normal_noise_std = 0,label = 0, bias = 0, dim =2):
    if couple: #two spirals
        assert BATCH_SIZE % 2 == 0, "the batch size should be an even number "
        BATCH_SIZE = BATCH_SIZE//2
        
    n = np.sqrt(np.random.rand(BATCH_SIZE,1),dtype='f') *540*(2*np.pi)/360
    d1x = -np.cos(n) * (n+bias) 
    d1y = -np.sin(n) * (n+bias) 
    
    data = (np.hstack((d1x, d1y)), np.hstack((d1x, d1y))) if couple else np.hstack((d1x, d1y))
    x = np.vstack(data)/12
    
    if normal_noise_std>0:
        x += np.random.randn(*x.shape) * normal_noise_std
    if unif_noise>0: 
        x += np.random.rand(*x.shape) * unif_noise
    if dim > 2: 
        x = random_manifold_updim(x, input_dim=2, out_dim = dim)
    return x.astype('float32')

def prepare_gaussian8_data(BATCH_SIZE=1000, scale = 0.1, dim=2, center_cotraction =0.7):
    centers = [(1, 0), (-1, 0), (0, 1), (0, -1),
               (1. / np.sqrt(2), 1. / np.sqrt(2)),
               (1. / np.sqrt(2), -1. / np.sqrt(2)),
               (-1. / np.sqrt(2), 1. / np.sqrt(2)),
               (-1. / np.sqrt(2), -1. / np.sqrt(2))]

    centers= np.array(centers)*center_cotraction
    colors = np.array(range(8)).astype('float64')
    #randeom sampling
    std_normal = np.random.randn(BATCH_SIZE,2)
    center_idx = np.random.choice(range(8), size=BATCH_SIZE, replace=True)
    data = std_normal*scale + centers[center_idx]
    colors = center_idx.astype('float64')
    
    if dim > 2: 
        data = random_manifold_updim(data, input_dim=2, out_dim = dim)

    return data.astype('float32'), colors
