import numba
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import time
import matplotlib.pyplot as plt
from iglo.manifold.neighbor_dataset import Neighbor_dataset
from iglo.manifold._util_manifold import random_nb_sparse,rand_seed, iglo_graph
from torch.utils.data import DataLoader
from iglo.evals.eval_tools import knn_clf_seq


def timer(e_time):
    hours, remainder = divmod(e_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
    #return hours, minutes, seconds

@numba.njit(locals={"norm": numba.float32})
def normalize_g_dist(g_dist, norm):
    return g_dist / (norm / 3)

@numba.njit
def compute_pairwise_distances(z_h):
    # This will compute the Euclidean distance between all pairs of points
    return np.sqrt(np.sum((z_h[:, None] - z_h[None, :]) ** 2, axis=2))

@numba.njit
def compute_neg_step(a, b, z_dist_square, diff, clamp):
    deno = (0.001 + z_dist_square) * (a * z_dist_square ** b + 1)
    neg_step = (2.0 * b) / (deno[:, :, None]) * diff
    return np.clip(neg_step, -clamp, clamp)

class iGLoMAP:
    def __init__(self,
                 n_neighbors=15,
                 ee=None,
                 a=1.57694,
                 b=0.8951,
                 initial_lr=1,
                 end_lr=0,
                 batch_size=100,
                 z_dim=2,
                 EPOCHS=None,
                 plot_freq=20,
                 seed=1234,
                 show=True,
                 vis_dir=None,
                 vis_s=1,
                 clamp=4,
                 lr_Q=0.01,
                 Z=None,
                 initial_tau=1,
                 end_tau=0.1,
                 exact_mu=True,
                 rainbow=False,
                 save_vis=False,
                 distance_normalization=True):

        self.n_neighbors = n_neighbors
        self.ee = 1 if ee is None else ee
        self.a = a
        self.b = b
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.EPOCHS = 150 if EPOCHS is None else EPOCHS
        self.plot_freq = plot_freq
        self.seed = seed
        self.show = show
        self.vis_dir = vis_dir
        self.vis_s = vis_s
        self.clamp = clamp
        self.lr_Q = lr_Q
        self.Z = Z
        self.initial_tau = initial_tau
        self.end_tau = end_tau
        self.exact_mu = exact_mu
        self.rainbow = rainbow
        self.save_Z = save_vis
        self.distance_normalization = distance_normalization
        self.Z_list = {}
        print("iGLoMAP initialized")

    @numba.njit(locals={"vm": numba.float32, "sig": numba.float32})
    def P_update(self, sig):
        P = np.exp(-self.g_dist / sig)
        np.fill_diagonal(P, 0)
        self.sparse_P = sparse.csr_matrix(P)
        del P

        v_i_dot = np.array(self.sparse_P.sum(axis=1)).flatten()
        vm = np.max(v_i_dot)
        v_i_dot /= vm
        self.v_i_dot = v_i_dot
        self.learning_ee = self.ee / vm



    def _fit_prepare(self, X, Y, precalc_graph=None, save_shortest_path=False, shortest_path_comp=True):
        if precalc_graph is None:
            g_dist = iglo_graph(X, self.n_neighbors, shortest_path_comp=shortest_path_comp)
        else:
            g_dist = precalc_graph

        if self.distance_normalization:
            data_1d = g_dist.reshape(-1)
            data_1d = data_1d[np.isfinite(data_1d)]
            norm = np.median(data_1d)
            print(norm)
            g_dist = self.normalize_g_dist(g_dist, norm)

        self.g_dist = g_dist
        self.P_update(sig=self.initial_tau)

        def random_idx_generator(idx):
            return random_nb_sparse(self.sparse_P, idx)

        indices = np.arange(X.shape[0]).reshape(-1, 1)
        indx_dataset_g = Neighbor_dataset(indices, neighbor_rule=random_idx_generator, return_idx=True)

        self.g_loader = DataLoader(indx_dataset_g, batch_size=self.batch_size, shuffle=True, num_workers=2)

        rand_seed(self.seed)

        if self.Z is None:
            self.Z = np.random.randn(X.shape[0], 2).astype(np.float32)
        else:
            assert self.Z.shape[0] == X.shape[0]
        self.Y = Y
        self.X = X.astype(np.float32)
        self.nu = self.sparse_P.todense()

    @numba.njit(locals={
        "z_dist_square": numba.float32[:, ::1],
        "neg_step": numba.float32[:, :, ::1]
    })
    def manual_single_negative_grad(self, z_h, a, b, idx_h):
        z_dist_square = compute_pairwise_distances(z_h)
        diff = -(z_h[:, None] - z_h[None, :])
        neg_step = compute_neg_step(a, b, z_dist_square, diff, self.clamp)
        if self.exact_mu:
            neg_step *= (1 - np.array(self.sparse_P[idx_h, :][:, idx_h].todense()))[:, :, None]
        return neg_step

    @numba.njit
    def manual_single_update(self, z_h, z_t, idx_h, alpha):
        neg_grad = self.manual_single_negative_grad(z_h, self.a, self.b, idx_h)
        neg_step = np.sum(neg_grad, axis=1)

        updated_z_h = z_h + alpha * self.learning_ee * neg_step

        znb_dist_square = np.sum((updated_z_h - z_t) ** 2, axis=1)
        grad_coeff = -2.0 * self.a * self.b * znb_dist_square ** (self.b - 1.0)
        grad_coeff /= self.a * znb_dist_square ** self.b + 1.0

        grad_coeff *= self.v_i_dot[idx_h]

        pos_step = grad_coeff[:, None] * (updated_z_h - z_t)
        pos_step = np.clip(pos_step, -self.clamp, self.clamp)
        tail_pos = -pos_step

        next_z_h = updated_z_h + alpha * pos_step
        next_z_t = z_t + alpha * tail_pos
        return next_z_h, next_z_t

    def _fit_particle_only(self):
        self.Z_list.update({0: self.Z.copy()})

        for epochs in range(self.EPOCHS):
            alpha = self.initial_lr - (self.initial_lr - self.end_lr) * (float(epochs) / float(self.EPOCHS))

            if (epochs > 5) and (epochs % 50 == 0):
                if self.initial_tau != self.end_tau:
                    sig = self.initial_tau - (self.initial_tau - self.end_tau) * (float(epochs) / float(self.EPOCHS)) ** 0.5
                    self.P_update(sig=sig)

            for ii, pack in enumerate(self.g_loader):
                idx_h, idx_t = pack
                z_h, z_t = self.Z[idx_h], self.Z[idx_t]
                z_h, z_t = self.manual_single_update(z_h, z_t, idx_h, alpha)
                self.Z[idx_h], self.Z[idx_t] = z_h, z_t

            if ((epochs + 1) % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS):
                if self.save_Z:
                    self.Z_list.update({epochs + 1: self.Z.copy()})
                if self.show:
                    self.vis(show=self.show, title=f"epoch {epochs+1}/{self.EPOCHS}",
                             path=None, s=self.vis_s, rainbow=self.rainbow, epochs=epochs + 1)

        if self.vis_dir is not None:
            path = f"{self.vis_dir}/z_list.dat"
            np.save(path, self.Z_list)

    def fit_transform(self, X, Y=None,precalc_graph=None, save_shortest_path = False, shortest_path_comp=True):
        begin = time.time()
        self._fit_prepare(X, Y, precalc_graph, save_shortest_path, shortest_path_comp)
        end = time.time()
        self.preparation_time = end - begin
        print(F"The learning is prepared, time spent:{timer(self.preparation_time)}")

        begin = time.time()
        self._fit_particle_only()
        end = time.time()
        self.fitting_time = end - begin
        print(F"GLoMAP fitting done, time spent: {timer(self.fitting_time)}")

        return self.Z

    def eval(self, Y, nns=[1, 2, 3, 4, 5, 10, 30]):
        Z_np = self.Z
        pairwise_Z = squareform(pdist(Z_np))

        cls = knn_clf_seq(40, pairwise_Z, Y)
        results = {}
        for K in nns:
            results[K] = cls(K)
        return results

    def vis(self, Y=None, axis=None, s=1, show=False, title=None, path=None, rainbow=False, close=True, epochs=None):
        Z0 = self.Z
        if axis is None:
            fig = plt.figure(figsize=(8, 8))
            axis = fig.add_subplot(111)
        if Y is None:
            axis.scatter(Z0[:, 0], Z0[:, 1], s=s)
        else:
            axis.scatter(Z0[:, 0], Z0[:, 1], c=Y, cmap='Spectral', s=s)
        axis.set_aspect('equal')
        axis.set_title(title)
        if path is not None:
            fig.savefig(path)
        if show:
            plt.show()
        if close:
            plt.close()

