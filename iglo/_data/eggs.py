import numpy as np

def egg_split(_min = -4, _max = 4, n=1000, r=1, eggs=1, square = True, alpha=1.0):
    
    A = (np.abs(_min) + np.abs(_max))**2
    
    if not square: 
        A *= 4

    n_unif = int(n * A * alpha /( alpha * A + eggs * (4-alpha)* np.pi* r**2))
    n_hsph = int(eggs * n * 4 *np.pi* r**2/(alpha*A+(4-alpha)*eggs*np.pi * r**2))

    return n_unif, n_hsph

def sample_spherical(npoints, dim=3, r=1,half=True):
    vec = np.random.randn(dim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    if half: 
        vec[dim-1] = np.abs(vec[dim-1])
    return vec*r

def unif_hole(n, _min=-4, _max=4, dim=3, z_val = 0, centers = [(0,0)], r=1, square=True):
    # so far, only for dim=3
    if square: 
        s1 = np.random.uniform(_min,_max, n)
    else: 
        s1 = np.random.uniform(_min*4,_max*4, n)
    s2 = np.random.uniform(_min,_max, n)
    s3 = np.ones(n)*z_val
    s = np.stack((s1,s2,s3)).reshape(dim,-1) # need to be fixed if general dim
    
    for x_c, y_c in centers:# need to be fixed if general dim
        remain = ((s[0]-x_c)**2 + (s[1]-y_c)**2)>(r**2)# need to be fixed if general dim
        s = s[:, remain]
    
    return s


    
def get_12_egg(n_unif=1000, n_egg=1000,_min = -4,_max = 4, dim=3,r =1, z_val = 0, color=True, seed = 1): 
    
    np.random.seed(seed+10)
    
    centers = ((-13,-2), (-13,2),(-8,-2), (-3,2), (2,+2),(7,-2), (12,2),(-8,+2), (-3,-2), (2,-2), (7,+2), (12,-2))    
    data = unif_hole(n_unif, _min,_max, dim, z_val, centers, r, square=False)
    for x_c, y_c in centers:
        vec = sample_spherical(int(n_egg/12), dim, r)
        data = np.vstack([data.transpose(), (vec+ np.array([x_c,y_c,z_val]).reshape(3,-1)).transpose()]).transpose()
    #data = data.transpose()
    if color: 
        color= data[2]
        return data, color
    else: 
        return data
