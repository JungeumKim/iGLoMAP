from IPython.core.debugger import set_trace
import torch 
import numpy as np
from scipy.sparse import coo_matrix
from numba import njit
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path


def rand_seed(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def np_pairwise_distances(X, dim=3, reduction="norm"):
    '''
        The input shape should be N x d
    '''
    n_data = X.shape[0]

    # Repeat X for pairwise difference calculation
    ref_c = np.tile(X.reshape(-1, dim), (n_data, 1)).reshape(n_data, n_data, dim)

    # Compute pairwise differences
    diff = ref_c - X.reshape(n_data, 1, dim)

    if reduction is None:
        return diff
    else:
        # Compute Euclidean norm along the specified dimension
        distances = np.linalg.norm(diff, axis=2)
        return distances

@njit
def compute_rescaled_dist_numba(knn_indices, knn_dists, sigmas, dist_scale=1.0):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]
    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    dists = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i

            index = i * n_neighbors + j
            rows[index] = i
            cols[index] = knn_indices[i, j]
            dists[index] = knn_dists[i, j] / min(sigmas[i], sigmas[knn_indices[i, j]])

    # Scale distances
    dists *= dist_scale
    return rows, cols, dists

# Usage of the Numba function
def compute_rescaled_dist(knn_indices, knn_dists, sigmas, dist_scale=1.0):
    rows, cols, dists = compute_rescaled_dist_numba(knn_indices, knn_dists, sigmas, dist_scale)

    # Create the sparse matrix outside the Numba function

    dmat = coo_matrix((dists, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))

    # Make it symmetric
    dists_symmetric = dmat.maximum(dmat.transpose()).tocsr()
    return dists_symmetric

def random_nb_dense(P, idx):
    n_P_transition = P[idx]
    m = P.shape[1]
    sum = n_P_transition.sum()
    if sum== 0:
        chosen_col = np.random.choice(m, 1).item()
    else:
        n_P_transition /= sum
        chosen_col = np.random.choice(m, 1, p=n_P_transition).item()
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
