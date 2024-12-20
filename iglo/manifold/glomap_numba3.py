from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
import numpy as np
import torch
from iglo.manifold.neighbor_dataset import Neighbor_dataset
from iglo.manifold._util_manifold_numba import np_pairwise_distances,random_nb_dense,rand_seed, iglo_graph
from torch.utils.data import DataLoader
import time
from numba import njit

def timer(e_time):
    hours, remainder = divmod(e_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
    #return hours, minutes, seconds

@njit
def P_update(g_dist,sig):
    P = np.exp(-g_dist/sig)
    np.fill_diagonal(P, 0)
    v_i_dot = P.sum(axis=1)
    vm = v_i_dot.max()
    v_i_dot /= vm
    return P, v_i_dot,vm

@njit
def normalize(g_dist):
    data_1d = g_dist.reshape(-1)
    data_1d = data_1d[np.isfinite(data_1d)]
    norm = np.median(data_1d)
    print(norm)
    g_dist = g_dist/(norm/3)
    return g_dist

class  iGLoMAP():
    def __init__(self,
                 n_neighbors = 15,
                 #d_thresh = np.inf,
                 ee = None,
                 a = 1.57694,
                 b = 0.8951,
                 show=True,
                 initial_lr = 1,
                 end_lr = 0,
                 batch_size = 100,
                 z_dim=2,
                 EPOCHS = None,
                 plot_freq = 20,
                 seed=1234,
                 clamp=4,
                 vis_s = 1,
                 device="cuda",
                 lr_Q = 0.01,
                 conv=False,
                 Q = None,
                 Z=None,
                 initial_tau=1,
                 end_tau =0.1,
                 exact_mu = True, rainbow=False, distance_normalization=True):


        ''' ARGUMENTS:
        optimizer: if None, manual gradient steps are used. else (e.g., sgd), then the SGD torch implementation used.
            But anyway because any momentum based method will not be used, either one should have no difference.
        n_neighbors: number of neighbors used to construct the graph for global geodesic distance estimates
        ee: the weight of the loss for the negative part
        device, lr_Q, conv: for the inductive modeling
        '''
        #self.d_thresh = d_thresh
        self.n_neighbors = n_neighbors
        self.ee = 1 if ee is None else ee
        self.a = a
        self.b = b
        self.initial_lr = initial_lr
        self.end_lr=end_lr
        self.batch_size = batch_size
        self.z_dim = z_dim
        if EPOCHS is None:
            self.EPOCHS =  300
        else:
            self.EPOCHS = EPOCHS
        self.plot_freq = plot_freq
        self.seed = seed
        self.vis_s = vis_s
        self.clamp= clamp
        self.device = device
        self.lr_Q = lr_Q
        self.conv= conv
        self.Q = Q
        self.Z = Z
        self.initial_tau = initial_tau
        self.end_tau= end_tau
        self.exact_mu = exact_mu
        self.rainbow=rainbow
        self.distance_normalization = distance_normalization
        self.show=show
        print("iGLoMAP initialized")



    def _fit_prepare(self, X,Y, precalc_graph=None, shortest_path_comp=True):
        rand_seed(self.seed)

        if precalc_graph is None:
            g_dist = iglo_graph(X, self.n_neighbors, shortest_path_comp = shortest_path_comp)
            #if save_shortest_path:
            #    self.shortest_path = np.copy(g_dist)
        else:
            g_dist = precalc_graph

        if self.distance_normalization:
            g_dist = normalize(g_dist)



        self.g_dist = g_dist
        self.P , self.v_i_dot, vm = P_update(self.g_dist, sig = self.initial_tau)
        self.learning_ee = self.ee/vm

        self.g_loader = self.get_loader(n=X.shape[0])


        if self.Z is None:
            self.Z = np.random.randn(X.shape[0], 2).astype(np.float32) #torch.randn(size=(X.shape[0],2), device = "cpu").float()
        else:
            assert self.Z.shape[0] == X.shape[0]

        self.Y = Y

        self.X = torch.from_numpy(X).float()


    def get_loader(self, n):

            def random_idx_generator(idx):
                return random_nb_dense(self.P, idx)

            indices = np.array(range(n)).reshape(-1, 1)

            indx_dataset_g = Neighbor_dataset(indices,
                                       neighbor_rule=random_idx_generator,
                                       return_idx=True)

            return DataLoader(indx_dataset_g,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  num_workers=2)

    def fit_transform(self, X, Y=None,precalc_graph=None,  shortest_path_comp=True):
        begin = time.time()
        self._fit_prepare(X, Y, precalc_graph,shortest_path_comp)
        end = time.time()
        self.preparation_time = end - begin
        print(F"The learning is prepared, time spent:{timer(self.preparation_time)}")

        begin = time.time()

        self._fit_particle_only()
        end = time.time()
        self.fitting_time = end - begin
        print(F"GLoMAP fitting done, time spent: {timer(self.fitting_time)}")


        return self.Z


    def manual_single_negative_grad(self,z_h,a,b,idx_h):
        z_dist_square = np_pairwise_distances(z_h, dim=2)**(2)
        deno = (0.001 + z_dist_square) * (a * z_dist_square**(b) + 1)

        grad_coeff = 2.0 * b
        grad_coeff = grad_coeff / deno
        # important : torch.pairwise_distances gives other - z_h. Therefore, should multiply by -1.
        diff = -np_pairwise_distances(z_h, dim=2, reduction=None)
        neg_step = np.expand_dims(grad_coeff, axis=2) * diff
        neg_step = np.clip(neg_step, -self.clamp, self.clamp) 
        if self.exact_mu:
            neg_step *= np.expand_dims(1-self.P[idx_h,:][:,idx_h],2)
        return neg_step

    def manual_single_update(self, z_h, z_t, idx_h,alpha):
        ## negative step
        neg_grad = self.manual_single_negative_grad(z_h,self.a, self.b, idx_h)
        neg_step = neg_grad.sum(axis=1)

        updated_z_h = z_h + alpha * self.learning_ee * neg_step

        ## positive step
        znb_dist_square = np.sum((updated_z_h - z_t) ** 2, axis=1) #(updated_z_h - z_t).norm(dim=1)**(2)
        grad_coeff = -2.0 * self.a * self.b * znb_dist_square**(self.b - 1.0)
        grad_coeff /= self.a * znb_dist_square**(self.b) + 1.0

        #set_trace() #check if view(-1) is unnecessary.
        grad_coeff *= self.v_i_dot[idx_h] #.clone().detach().view(-1)

        # pos_step = grad_coeff.view(-1,1) * (z_h-z_t)
        pos_step = grad_coeff.reshape(-1, 1) * (updated_z_h - z_t)
        pos_step = np.clip(pos_step, -self.clamp, self.clamp) #pos_step.clamp(-self.clamp, self.clamp)
        tail_pos = -pos_step

        next_z_h = updated_z_h + alpha * pos_step
        next_z_t = z_t + alpha * tail_pos
        return next_z_h, next_z_t



    def _fit_particle_only(self):

        for epochs in range(self.EPOCHS):
            alpha = self.initial_lr - (self.initial_lr-self.end_lr) * (float(epochs) / float(self.EPOCHS))#mannual step size.

            if (epochs>5) and (epochs % 50 == 0):
                if (self.initial_tau !=self.end_tau):
                    sig = self.initial_tau - (self.initial_tau-self.end_tau) * (float(epochs) / float(self.EPOCHS))**0.5
                    self.P , self.v_i_dot, vm = P_update(self.g_dist, sig = sig)
                    self.learning_ee = self.ee/vm

            #early = (epochs < self.EPOCHS*0.3)
            for ii, pack in enumerate(self.g_loader):
                _, idx_h, _, idx_t = pack
                z_h, z_t = self.Z[idx_h.long()], self.Z[idx_t.long()]
                z_h, z_t = self.manual_single_update(z_h, z_t, idx_h,alpha)#,early)
                self.Z[idx_h.long()], self.Z[idx_t.long()] = z_h,z_t#.astype(np.float32)

            if self.show:
                if (((epochs+1) % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS)):
                    self.vis(show=self.show,title=F"epoch{epochs+1}/{self.EPOCHS}",
                        path=None,s=self.vis_s, rainbow=self.rainbow,epochs = epochs+1)


    def vis(self,Y=None,axis=None, s=1, show=False,title=None,path=None,rainbow=False, close=True, epochs = None):

        if Y is not None:
            color = Y
        elif self.Y is not None:
            color = self.Y
        else:
            color=None

        if axis is not None:
            assert path is None, "when axis is given, we cannot save it."

        else:
            fig = plt.figure(figsize=(8, 8))
            axis = fig.add_subplot(111)

        Z0 = self.Z

        if color is None:
            axis.scatter(Z0[:, 0], Z0[:, 1], s=s)
        else:
            if rainbow:
                axis.scatter(Z0[:, 0], Z0[:, 1], c=color, cmap=plt.cm.gist_rainbow, s=s)
            else:
                axis.scatter(Z0[:, 0], Z0[:, 1], c=color, cmap=plt.cm.Spectral, s=s)
        axis.set_aspect('equal')
        axis.set_title(title)
        if path is not None:
            fig.savefig(path)
        if show:
            plt.show()
        if close:
            plt.close()



