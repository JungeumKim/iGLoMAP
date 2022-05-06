import numpy as np


def hierarch_data2(n, seed=1, macro_seed=1, meso_seed=1, micro_seed=1, dim=20):
    assert n % 125 == 0
    k = int(n / 125)

    np.random.seed(macro_seed)

    macro_centers = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.diag(np.ones(dim) * 1000), size=5)

    np.random.seed(meso_seed)
    meso_centers = []
    for i in range(5):
        meso_center = np.random.multivariate_normal(mean=macro_centers[i], cov=np.diag(np.ones(dim) * 100), size=5)
        meso_centers.append(meso_center)
    meso_centers = np.vstack(meso_centers)

    np.random.seed(micro_seed)
    micro_centers = []
    for i in range(25):
        micro_center = np.random.multivariate_normal(mean=meso_centers[i], cov=np.diag(np.ones(dim) * 10), size=5)
        micro_centers.append(micro_center)
    micro_centers = np.vstack(micro_centers)

    np.random.seed(seed)
    data = []
    for i in range(125):
        point = np.random.multivariate_normal(mean=micro_centers[i], cov=np.diag(np.ones(dim) * 1), size=k)
        data.append(point)
    data = np.vstack(data)

    c_macro = []
    for i in range(5):
        c_macro.append(np.tile(i, int(n / 5)).reshape(-1))
    c_macro = np.concatenate(c_macro)

    c_meso = []
    for i in range(25):
        c_meso.append(np.tile(i, int(n / 25)).reshape(-1))
    c_meso = np.concatenate(c_meso)

    c_micro = []
    for i in range(125):
        c_micro.append(np.tile(i, int(n / 125)).reshape(-1))
    c_micro = np.concatenate(c_micro)


    return data, c_macro, c_meso, c_micro

def hierarch_data(n, seed=1, macro_seed = 1, meso_seed = 1, micro_seed = 1,add_cover=False): 
    assert n%125 ==0
    k = int(n/125)
    
    np.random.seed(macro_seed)
    
    macro_centers = np.random.multivariate_normal(mean=np.zeros(50), cov = np.diag(np.ones(50)*10000), size = 5)
    
    np.random.seed(meso_seed)
    meso_centers = []
    for i in range(5):
        meso_center = np.random.multivariate_normal(mean=macro_centers[i], cov = np.diag(np.ones(50)*1000), size = 5)
        meso_centers.append(meso_center)
    meso_centers = np.vstack(meso_centers)

    np.random.seed(micro_seed)
    micro_centers = []
    for i in range(25):
        micro_center = np.random.multivariate_normal(mean=meso_centers[i], cov = np.diag(np.ones(50)*100), size = 5)
        micro_centers.append(micro_center)
    micro_centers = np.vstack(micro_centers)

    np.random.seed(seed)
    data = []
    for i in range(125):
        point = np.random.multivariate_normal(mean=micro_centers[i], cov = np.diag(np.ones(50)*10), size = k)
        data.append(point)
    data = np.vstack(data)
    
    c_macro = []
    for i in range(5):
        c_macro.append(np.tile(i, int(n/5)).reshape(-1))
    c_macro = np.concatenate(c_macro)

    c_meso = []
    for i in range(25):
        c_meso.append(np.tile(i, int(n/25)).reshape(-1))
    c_meso = np.concatenate(c_meso)

    c_micro= []
    for i in range(125):
        c_micro.append(np.tile(i, int(n/125)).reshape(-1))
    c_micro = np.concatenate(c_micro)
    
    if add_cover: 
        s = np.random.randn(n, 50)
        s = 285 * s / np.sqrt(np.sum(s ** 2, 1)[:, None])
        
        data = np.vstack([data,s])
        c_macro = np.concatenate([c_macro, np.repeat(6,n)])
        c_meso  = np.concatenate([c_meso,  np.repeat(26,n)])
        c_micro = np.concatenate([c_micro, np.repeat(126,n)])
        
    return data, c_macro, c_meso, c_micro