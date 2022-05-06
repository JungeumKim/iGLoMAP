# modified from TOPO_AE (Moor at all, 2020 ICML)
# modification: self.pairwise_X now is not only L2, but also global,rescaled_global. See, MeasureCalculator.__init__

import umap as umap_  # import takes a while
from sklearn.utils.graph import graph_shortest_path
from sklearn.utils import check_random_state
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import itertools
from encodingGAN.evals.locals import local_msrs_test, local_msrs_np
from sklearn.neighbors import NearestNeighbors
from encodingGAN.manifold._util_manifold import compute_rescaled_dist

class MeasureRegistrator():
    """Keeps track of measurements in Measure Calculator."""
    k_independent_measures = {}
    k_dependent_measures = {}

    def register(self, is_k_dependent):
        def k_dep_fn(measure):
            self.k_dependent_measures[measure.__name__] = measure
            return measure

        def k_indep_fn(measure):
            self.k_independent_measures[measure.__name__] = measure
            return measure

        if is_k_dependent:
            return k_dep_fn
        return k_indep_fn

    def get_k_independent_measures(self):
        return self.k_independent_measures

    def get_k_dependent_measures(self):
        return self.k_dependent_measures


class MeasureCalculator():
    measures = MeasureRegistrator()

    def __init__(self, k_max, X, Z, Y, 
                 X_test=None, Z_test=None, Y_test=None, 
                 distance="L2",local=True,
                 knn_n_neighbors=[1, 3, 5, 10, 15, 20, 25, 30], dataset=None, PP=15,pairwise_X=None):

        self.k_max = k_max
        self.PP=PP
        
        self.knn_n_neighbors=knn_n_neighbors
        if pairwise_X is not None:
            self.pairwise_X = pairwise_X
        elif "egg" in dataset:
            true_ebb = X[:, :2]
            self.pairwise_X = squareform(pdist(true_ebb))
        elif dataset == "scurve":
            generator = check_random_state(1)
            t = 3 * np.pi * (generator.rand(1, X.shape[0]) - 0.5)
            y = 2.0 * generator.rand(1, X.shape[0])
            true_ebb = np.concatenate((t.reshape(1, -1), y.reshape(1, -1))).transpose()
            self.pairwise_X = squareform(pdist(true_ebb))
        elif dataset == "severe": #only n=6000 will work. Remember.
            random_state = check_random_state(1)
            p = random_state.rand(6000) * (2 * np.pi - 0.55)
            t = random_state.rand(6000) * np.pi
            indices = ((t < (np.pi - (np.pi / 8))) & (t > ((np.pi / 8))))
            true_ebb = np.concatenate((t.reshape(1, -1), p.reshape(1, -1))).transpose()[indices]
            self.pairwise_X = squareform(pdist(true_ebb))
        else:
            if distance == "L2":
                self.pairwise_X = squareform(pdist(X))
            elif distance == "global":
                self.pairwise_X = graph_shortest_path(squareform(pdist(X)))
            elif distance == "rescaled_global":
                #umap = umap_.UMAP(transform_mode="graph", densmap=True)
                #umap.fit_transform(X)
                #self.pairwise_X = graph_shortest_path(umap.graph_dists_.tocsr())
                nbrs = NearestNeighbors(n_neighbors=self.PP).fit(X)
                knn_dists, knn_indices = nbrs.kneighbors(X)  # the first column is with the point itself.
                sigmas = knn_dists.sum(axis=1) / (self.PP - 1)
                rescaled_knn_dists_mat = compute_rescaled_dist(knn_indices, knn_dists, sigmas, 1, True)
                self.pairwise_X = graph_shortest_path(rescaled_knn_dists_mat, directed=False)

            else:
                assert False, "valid distance argument is [L2,global,rescaled_global]"


        self.pairwise_Z = squareform(pdist(Z))
        self.neighbours_X, self.ranks_X = \
            self._neighbours_and_ranks(self.pairwise_X, k_max)
        self.neighbours_Z, self.ranks_Z = \
            self._neighbours_and_ranks(self.pairwise_Z, k_max)

        self.data = [Z,Y]
        self.data_test = [Z_test,Y_test]
        self.local=local

    @staticmethod
    def _neighbours_and_ranks(distances, k):
        """
        Inputs:
        - distances,        distance matrix [n times n],
        - k,                number of nearest neighbours to consider
        Returns:
        - neighbourhood,    contains the sample indices (from 0 to n-1) of kth nearest neighbor of current sample [n times k]
        - ranks,            contains the rank of each sample to each sample [n times n], whereas entry (i,j) gives the rank that sample j has to i (the how many 'closest' neighbour j is to i)
        """
        # Warning: this is only the ordering of neighbours that we need to
        # extract neighbourhoods below. The ranking comes later!
        indices = np.argsort(distances, axis=-1, kind='stable')

        # Extract neighbourhoods.
        neighbourhood = indices[:, 1:k + 1]

        # Convert this into ranks (finally)
        ranks = indices.argsort(axis=-1, kind='stable')

        return neighbourhood, ranks

    def get_X_neighbours_and_ranks(self, k):
        return self.neighbours_X[:, :k], self.ranks_X

    def get_Z_neighbours_and_ranks(self, k):
        return self.neighbours_Z[:, :k], self.ranks_Z

    def compute_k_independent_measures(self):
        return {key: fn(self) for key, fn in
                self.measures.get_k_independent_measures().items()}

    def compute_k_dependent_measures(self, k):
        return {key: fn(self, k) for key, fn in
                self.measures.get_k_dependent_measures().items()}

    def compute_measures_for_ks(self, ks):
        return {
            key: np.array([fn(self, k) for k in ks])
            for key, fn in self.measures.get_k_dependent_measures().items()
        }

    @measures.register(False)
    def stress(self):
        sum_of_squared_differences = \
            np.square(self.pairwise_X - self.pairwise_Z).sum()
        sum_of_squares = np.square(self.pairwise_Z).sum()

        return np.sqrt(sum_of_squared_differences / sum_of_squares)

    @measures.register(False)
    def rmse(self):
        n = self.pairwise_X.shape[0]
        sum_of_squared_differences = np.square(
            self.pairwise_X - self.pairwise_Z).sum()
        return np.sqrt(sum_of_squared_differences / n ** 2)

    @staticmethod
    def _trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                         Z_ranks, n, k):
        '''
        Calculates the trustworthiness measure between the data space `X`
        and the latent space `Z`, given a neighbourhood parameter `k` for
        defining the extent of neighbourhoods.
        '''

        result = 0.0

        # Calculate number of neighbours that are in the $k$-neighbourhood
        # of the latent space but not in the $k$-neighbourhood of the data
        # space.
        for row in range(X_ranks.shape[0]):
            missing_neighbours = np.setdiff1d(
                Z_neighbourhood[row],
                X_neighbourhood[row]
            )

            for neighbour in missing_neighbours:
                result += (X_ranks[row, neighbour] - k)

        return 1 - 2 / (n * k * (2 * n - 3 * k - 1)) * result

    @measures.register(True)
    def trustworthiness(self, k):
        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        return self._trustworthiness(X_neighbourhood, X_ranks, Z_neighbourhood,
                                     Z_ranks, n, k)

    @measures.register(True)
    def continuity(self, k):
        '''
        Calculates the continuity measure between the data space `X` and the
        latent space `Z`, given a neighbourhood parameter `k` for setting up
        the extent of neighbourhoods.

        This is just the 'flipped' variant of the 'trustworthiness' measure.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)
        n = self.pairwise_X.shape[0]
        # Notice that the parameters have to be flipped here.
        return self._trustworthiness(Z_neighbourhood, Z_ranks, X_neighbourhood,
                                     X_ranks, n, k)

    @measures.register(True)
    def rank_correlation(self, k):
        '''
        Calculates the spearman rank correlation of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]
        # we gather
        gathered_ranks_x = []
        gathered_ranks_z = []
        for row in range(n):
            # we go from X to Z here:
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]
                gathered_ranks_x.append(rx)
                gathered_ranks_z.append(rz)
        rs_x = np.array(gathered_ranks_x)
        rs_z = np.array(gathered_ranks_z)
        coeff, _ = spearmanr(rs_x, rs_z)

        ##use only off-diagonal (non-trivial) ranks:
        # inds = ~np.eye(X_ranks.shape[0],dtype=bool)
        # coeff, pval = spearmanr(X_ranks[inds], Z_ranks[inds])
        return coeff

    @measures.register(True)
    def mrre(self, k):
        '''
        Calculates the mean relative rank error quality metric of the data
        space `X` with respect to the latent space `Z`, subject to its $k$
        nearest neighbours.
        '''

        X_neighbourhood, X_ranks = self.get_X_neighbours_and_ranks(k)
        Z_neighbourhood, Z_ranks = self.get_Z_neighbours_and_ranks(k)

        n = self.pairwise_X.shape[0]

        # First component goes from the latent space to the data space, i.e.
        # the relative quality of neighbours in `Z`.

        mrre_ZX = 0.0
        for row in range(n):
            for neighbour in Z_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                mrre_ZX += abs(rx - rz) / rz

        # Second component goes from the data space to the latent space,
        # i.e. the relative quality of neighbours in `X`.

        mrre_XZ = 0.0
        for row in range(n):
            # Note that this uses a different neighbourhood definition!
            for neighbour in X_neighbourhood[row]:
                rx = X_ranks[row, neighbour]
                rz = Z_ranks[row, neighbour]

                # Note that this uses a different normalisation factor
                mrre_XZ += abs(rx - rz) / rx

        # Normalisation constant
        C = n * sum([abs(2 * j - n - 1) / j for j in range(1, k + 1)])
        return mrre_ZX / C, mrre_XZ / C




    @measures.register(False)
    def correlation(self):
        corr = np.corrcoef(squareform(self.pairwise_Z), squareform(self.pairwise_X))[0, 1]
        return corr


    def run(self, k_min=10, k_max=200, k_step=10, single=True):

        ks = list(range(k_min, k_max + k_step, k_step))

        indep_measures = self.compute_k_independent_measures()
        dep_measures = self.compute_measures_for_ks(ks)
        mean_dep_measures = {
            'mean_' + key: values.mean() for key, values in dep_measures.items()
        }

        if single:
            return {
                key: value for key, value in
                itertools.chain(indep_measures.items(), mean_dep_measures.items())
            }
        else:
            return {
                key: value for key, value in
                itertools.chain(indep_measures.items(),
                                mean_dep_measures.items(),
                                dep_measures.items())
            }