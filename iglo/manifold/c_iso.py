from IPython.core.debugger import set_trace
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
import time
from sklearn.manifold import MDS


def c_compute_rescaled_dist(
        knn_indices, knn_dists, Ms):
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
            dists[i * n_neighbors + j] = knn_dists[i, j] / np.sqrt(Ms[i]*Ms[knn_indices[i, j]]) # f_max version
    print("rescale KNN_C_iso")
    dmat = coo_matrix((dists, (rows, cols)), shape=(n_samples,n_samples))
    dists = dmat.maximum(dmat.transpose()).tocsr()  # to symmetric
    return dists

def C_iso_graph(X, n_neighbors):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X)
    knn_dists, knn_indices = nbrs.kneighbors(X)  # the first column is with the point itself.
    print(F"Nearest Graph Done")
    Ms = knn_dists.sum(axis=1) / (n_neighbors)
    rescaled_knn_dists_mat = c_compute_rescaled_dist(knn_indices, knn_dists, Ms)
    print(F"Rescale of C-isomap Done")
    shortest= shortest_path(rescaled_knn_dists_mat, directed=False, return_predecessors=False)
    print(F"shortest path search of C-isomap Done")
    return shortest



def timer(e_time):
    hours, remainder = divmod(e_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
    #return hours, minutes, seconds

class  C_isomap():
    def __init__(self, n_neighbors = 15):

        self.n_neighbors = n_neighbors

    def fit_transform(self, X,Y, shortest_path_comp=True):
        begin = time.time()
        g_dist = C_iso_graph(X, self.n_neighbors, shortest_path_comp = shortest_path_comp)
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        embedding = mds.fit_transform(g_dist)
        self.embedding = embedding
        end = time.time()
        self.fitting_time = end - begin
        print(F"C_isomap fitting done, time spent: {timer(self.fitting_time)}")
        return embedding

