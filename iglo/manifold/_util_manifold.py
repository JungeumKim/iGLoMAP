from IPython.core.debugger import set_trace
import torch 
import numpy as np
from scipy.sparse import coo_matrix

from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path


def rand_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def torch_pairwise_distances(X, dim =3,grad_stop=False,reduction="norm"):
    '''
        At this moment, the input shape should be N x d
    '''
    n_data = X.shape[0]
    
    ref_c =torch.stack([X.view(-1,  dim)] * n_data, dim=0)
    if grad_stop:
        ref_c = ref_c.detach()
    diff = ref_c-X.view(-1,1,dim)
    if reduction is None:
        return diff
    else:
        distances = diff.norm(dim=2)
        return distances

#modified from umap
def compute_rescaled_dist(
        knn_indices, knn_dists, sigmas, dist_scale=1.0):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    dists = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            dists[i * n_neighbors + j] = knn_dists[i, j] / min(sigmas[i], sigmas[knn_indices[i, j]]) # f_max version
    print("rescale KNN_JK")
    dists *=  dist_scale
    dmat = coo_matrix((dists, (rows, cols)), shape=(n_samples,n_samples))
    dists = dmat.maximum(dmat.transpose()).tocsr()  # to symmetric
    print("JK_local_construction: done")
    return dists

def compute_nu(knn_indices, knn_dists, sigmas, dist_scale=1.0, offset=False):
    print("umap style local construction begins")

    knn_indices, knn_dists = knn_indices[:,1:], knn_dists[:,1:] # the first column is with itself. Don't need it.

    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1] #the original n_neighbor.
    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
    if offset:
        rhos = knn_dists.min(axis=1)
        knn_dists = np.clip(knn_dists - rhos.reshape(-1,1), a_min=0, a_max = None)
        sigmas = knn_dists.sum(axis=1)/(n_neighbors-1)
    umap_dists = dist_scale*(knn_dists/sigmas.reshape(-1,1))
    p = np.exp(-umap_dists)
    p = coo_matrix((p.flatten(), (rows, cols)), shape=(n_samples, n_samples))#.tocsr()
    transpose = p.transpose()
    prod_matrix = p.multiply(transpose)
    nu = p + transpose - prod_matrix
    nu.eliminate_zeros()
    print("umap_style_local_construction: done")
    return nu


def random_nb_sparse(P, idx):
    # number of neighbors = number of non-zero columns of idx row
    n_nb = P.indptr[idx + 1] - P.indptr[idx]

    # column idices of the neighbors of idx row
    nb = P.indices[P.indptr[idx]:P.indptr[idx + 1]]

    n_P_transition = P.data[P.indptr[idx]:P.indptr[idx + 1]]

    # normalization within n_nb probs.
    summ=n_P_transition.sum()
    if summ>0:
        n_P_transition /= n_P_transition.sum()
    # random selection
        chosen_ptr = np.random.choice(n_nb, 1, p=n_P_transition).item()
        chosen_col = nb[chosen_ptr]
    else:
        chosen_col = np.random.choice(P.shape[0], 1).item()

    return chosen_col


def iglo_graph(X, n_neighbors, shortest_path_comp=True):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    knn_dists, knn_indices = nbrs.kneighbors(X)  # the first column is with the point itself.
    #sigmas = knn_dists.sum(axis=1) / (n_neighbors)
    sigmas = np.sqrt(np.power(knn_dists,2).sum(axis=1) / (n_neighbors))
    print(F"DTM r=2, empirical version used, data size: n={sigmas.shape[0]}.")
    rescaled_knn_dists_mat = compute_rescaled_dist(knn_indices, knn_dists, sigmas, 1)

    if shortest_path_comp:
        return shortest_path(rescaled_knn_dists_mat, directed=False, return_predecessors=False)
    else:
        return rescaled_knn_dists_mat.toarray()
