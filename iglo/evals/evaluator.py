# modified from TOPO_AE (Moor at all, 2020 ICML)
# modification: self.pairwise_X now is not only L2, but also global,rescaled_global. See, MeasureCalculator.__init__

import numpy as np
from scipy.spatial.distance import pdist, squareform
import itertools


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

    def __init__(self,  X, Z):



        self.pairwise_X = squareform(pdist(X))

        self.pairwise_Z = squareform(pdist(Z))


    def compute_k_independent_measures(self):
        return {key: fn(self) for key, fn in
                self.measures.get_k_independent_measures().items()}

    @measures.register(False)
    def density_kl_global(self, sigma=0.1):
        X = self.pairwise_X
        X = X / X.max()
        Z = self.pairwise_Z
        Z = Z / Z.max()

        density_x = np.sum(np.exp(-(X ** 2) / sigma), axis=-1)
        density_x /= density_x.sum(axis=-1)

        density_z = np.sum(np.exp(-(Z ** 2) / sigma), axis=-1)
        density_z /= density_z.sum(axis=-1)

        return (density_x * (np.log(density_x) - np.log(density_z))).sum()

    @measures.register(False)
    def density_kl_global_10(self):
        return self.density_kl_global(10.)

    @measures.register(False)
    def density_kl_global_1(self):
        return self.density_kl_global(1.)

    @measures.register(False)
    def density_kl_global_01(self):
        return self.density_kl_global(0.1)

    @measures.register(False)
    def density_kl_global_001(self):
        return self.density_kl_global(0.01)

    @measures.register(False)
    def density_kl_global_0001(self):
        return self.density_kl_global(0.001)

    @measures.register(False)
    def correlation(self):
        corr = np.corrcoef(squareform(self.pairwise_Z), squareform(self.pairwise_X))[0, 1]
        return corr


    def run(self, k_min=10, k_max=200, k_step=10, single=True):

        #ks = list(range(k_min, k_max + k_step, k_step))

        indep_measures = self.compute_k_independent_measures()
        #dep_measures = self.compute_measures_for_ks(ks)


        return {
                key: value for key, value in
                itertools.chain(indep_measures.items())#,
                                #dep_measures.items())
        }
