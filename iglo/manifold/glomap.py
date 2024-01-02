from IPython.core.debugger import set_trace

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import torch
from iglo.manifold.neighbor_dataset import Neighbor_dataset
from iglo.manifold._util_manifold import torch_pairwise_distances,random_nb_sparse,rand_seed, iglo_graph
from torch.utils.data import DataLoader
from iglo._data import get_precalc_graph
from iglo.evals.eval_tools import knn_clf_seq
from scipy.spatial.distance import pdist, squareform
# model: to initialize
import iglo._networks.network_conv_grey as ncg
import iglo._networks.network_toy as nt
from os.path import join
import copy
import time

def timer(e_time):
    hours, remainder = divmod(e_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h:{int(minutes)}m:{int(seconds)}s"
    #return hours, minutes, seconds

class  iGLoMAP():
    def __init__(self,
                 n_neighbors = 15,
                 #d_thresh = np.inf,
                 ee = None,
                 a = 1.57694,
                 b = 0.8951,
                 initial_lr = 1,
                 end_lr = 0,
                 batch_size = 100,
                 z_dim=2,
                 EPOCHS = 300,
                 plot_freq = 20,
                 seed=1234,
                 show=True,
                 vis_dir = None,
                 vis_s = 1,
                 clamp=4,
                 device="cuda",
                 lr_Q = 0.01,
                 conv=False,
                 Q = None,
                 use_mapper=True,
                 Z=None,
                 initial_sigma=1,
                 end_sigma =1,
                 initial_tau_percentile=None, #75
                 end_tau_percentile = None, #15
                 exact_mu = True, rainbow=False, save_vis=False):

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
        self.EPOCHS = EPOCHS
        self.plot_freq = plot_freq
        self.seed = seed
        self.show = show
        self.vis_dir = vis_dir
        self.vis_s = vis_s
        self.clamp= clamp
        self.device = device
        self.lr_Q = lr_Q
        self.conv= conv
        self.Q = Q
        self.use_mapper = use_mapper
        self.Z = Z
        self.initial_sigma = initial_sigma
        self.end_sigma= end_sigma
        self.initial_tau_percentile = initial_tau_percentile
        self.end_tau_percentile = end_tau_percentile
        self.sigma_scaled=False
        self.exact_mu = exact_mu
        self.rainbow=rainbow
        self.save_Z = save_vis

        if self.save_Z:
            self.Z_list = {}
        print("iGLoMAP initialized")

    def P_update(self,sig):

        #g_dist[g_dist == 0] = float("inf")
        P = np.exp(-self.g_dist/sig)
        np.fill_diagonal(P, 0)
        #del g_dist
        self.sparse_P = sparse.csr_matrix(P)
        del P

        v_i_dot = self.sparse_P.sum(axis=1)
        v_i_dot = torch.from_numpy(v_i_dot)
        vm = v_i_dot.max()
        v_i_dot /= vm
        self.v_i_dot = v_i_dot
        self.learning_ee = self.ee/vm

    def _fit_prepare(self, X,Y, precalc_graph=None, save_shortest_path = False, shortest_path_comp=True):

        if precalc_graph is None:
            g_dist = iglo_graph(X, self.n_neighbors, shortest_path_comp = shortest_path_comp)
            #if save_shortest_path:
            #    self.shortest_path = np.copy(g_dist)
        else:
            g_dist = precalc_graph
        #set_trace()
        #g_dist[g_dist > self.d_thresh] = float("inf")
        self.g_dist = g_dist
        if self.end_tau_percentile is not None:
            if not self.sigma_scaled:
                self.percentiles = np.percentile(self.g_dist.reshape(-1), [self.end_tau_percentile,self.initial_tau_percentile])
                self.initial_sigma = self.initial_sigma/self.percentiles[1]
                self.end_sigma= self.end_sigma/self.percentiles[0]
                self.sigma_scaled=True
        self.P_update(sig = self.initial_sigma)

        def random_idx_generator(idx):
            return random_nb_sparse(self.sparse_P, idx)
        indices = np.array(range(X.shape[0])).reshape(-1, 1)
        indx_dataset_g = Neighbor_dataset(indices,
                                   neighbor_rule=random_idx_generator,
                                   return_idx=True)

        self.g_loader = DataLoader(indx_dataset_g,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=2)

        rand_seed(self.seed)

        if self.use_mapper:
            if self.Q is None:
                if self.conv:
                    Q = ncg.Q_MNIST_BN(device="cpu", z_dim=self.z_dim,
                               layer_factor=16)
                else:
                    Q = nt.Q_2dim(device=self.device, leaky=0.01, z_dim=self.z_dim,
                              dim=X.shape[1], factor=128)
                    
            else:
                Q = self.Q
            Q = Q.to(self.device)
            with torch.no_grad():
                self.Z = Q(torch.from_numpy(X).float().to(self.device))

            self.Q = Q.to(self.device)

        else:
            if self.Z is None:
                self.Z = torch.randn(size=(X.shape[0],2), device = "cpu").float()
            else:
                assert self.Z.shape[0] == X.shape[0]
        self.Y = Y

        self.X = torch.from_numpy(X).float()

        self.nu = torch.tensor(self.sparse_P.todense())

    def fit_transform(self, X, Y=None,precalc_graph=None, eval=True,save_shortest_path = False, shortest_path_comp=True):
        begin = time.time()
        self._fit_prepare(X, Y, precalc_graph, save_shortest_path, shortest_path_comp)
        end = time.time()
        self.preparation_time = end - begin
        print(F"The learning is prepared, time spent:{timer(self.preparation_time)}")

        begin = time.time()
        if self.use_mapper:
            self._fit_particle()
        else:
            self._fit_particle_only()
        end = time.time()
        self.fitting_time = end - begin
        print(F"iGLoMAP fitting done, time spent: {timer(self.fitting_time)}")

        #set_trace()
        if (eval) and (not  isinstance(Y, type(None))):
            self.eval_result = self.eval(Y)
            print("eval for knns:",self.eval_result)

        return self.Z

    def get_Z(self,X):
        Z0 = []
        with torch.no_grad():
            for i in range(int(X.shape[0] / self.batch_size)):
                z = self.Q(X[i * self.batch_size: (i + 1) * self.batch_size].to(self.device))
                Z0.append(z.cpu())
            if (X.shape[0] % self.batch_size) > 0:
                i = i + 1
                z = self.Q(X[i * self.batch_size:].to(self.device))
                Z0.append(z.cpu())
        Z = torch.cat(Z0)
        return Z

    def generalization(self, X, Y = None, plot = True, axis=None, s=1, title=None,path = None,
                       show=True,close=True, epochs = None):
        if isinstance(X,np.ndarray):
            X = torch.tensor(X).float()
        Z0 = self.get_Z(X).numpy()
        if plot:
            color = Y
            if axis is None:
                fig = plt.figure(figsize=(8, 8))
                axis = fig.add_subplot(111)
            else:
                assert path is None, "when axis is given, we cannot save it."
            if color is None:
                axis.scatter(Z0[:, 0], Z0[:, 1], s=s)
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

        return Z0



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

        if self.use_mapper:
            self.Z = self.get_Z(self.X)

        Z0 = self.Z.numpy()

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
        return copy.copy(Z0)


    def manual_single_negative_grad(self,z_h,a,b,idx_h):
        z_dist_square = torch_pairwise_distances(z_h, dim=2).pow(2)
        deno = (0.001 + z_dist_square) * (a * z_dist_square.pow(b) + 1)

        grad_coeff = 2.0 * b
        grad_coeff = grad_coeff / deno
        # important : torch.pairwise_distances gives other - z_h. Therefore, should multiply by -1.
        diff = -torch_pairwise_distances(z_h, dim=2, reduction=None)
        neg_step = grad_coeff.unsqueeze(2) * diff
        neg_step = neg_step.clamp(-self.clamp, self.clamp)
        if self.exact_mu:
            neg_step *= np.expand_dims(1-self.sparse_P[idx_h,:][:,idx_h].todense(),2)
        return neg_step
    def manual_single_update(self, z_h, z_t, idx_h,alpha):
        ## negative step
        neg_grad = self.manual_single_negative_grad(z_h,self.a, self.b, idx_h)
        neg_step = neg_grad.sum(dim=1)

        updated_z_h = z_h + alpha * self.learning_ee * neg_step

        ## positive step
        znb_dist_square = (updated_z_h - z_t).norm(dim=1).pow(2)
        grad_coeff = -2.0 * self.a * self.b * znb_dist_square.pow(self.b - 1.0)
        grad_coeff /= self.a * znb_dist_square.pow(self.b) + 1.0

        grad_coeff *= self.v_i_dot[idx_h].clone().detach().view(-1)

        # pos_step = grad_coeff.view(-1,1) * (z_h-z_t)
        pos_step = grad_coeff.view(-1, 1) * (updated_z_h - z_t)
        pos_step = pos_step.clamp(-self.clamp, self.clamp)
        tail_pos = -pos_step

        next_z_h = updated_z_h + alpha * pos_step
        next_z_t = z_t + alpha * tail_pos
        return next_z_h, next_z_t

    def _fit_particle(self):
        self.Z_list.update({0:copy.copy(self.Z)})

        optim = torch.optim.Adam(self.Q.parameters(), lr=self.lr_Q)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
        for epochs in range(self.EPOCHS):
            alpha = self.initial_lr - (self.initial_lr-self.end_lr) * (float(epochs) / float(self.EPOCHS)) #mannual step size.

            if (epochs>5) and (epochs % 50 == 0):
                if (self.initial_sigma !=self.end_sigma):
                    sig = self.initial_sigma - (self.initial_sigma-self.end_sigma) * (float(epochs) / float(self.EPOCHS)) #**(2) #m
                    self.P_update(sig = sig)

            #early = (epochs < self.EPOCHS*0.3)
            for ii, pack in enumerate(self.g_loader):

                idx_h_dum, idx_h, idx_t_dum, idx_t = pack

                x_h, x_t = self.X[idx_h.long()].to(self.device), self.X[idx_t.long()].to(self.device)
                z_h_Q, z_t_Q = self.Q(x_h), self.Q(x_t)
                z_h, z_t = z_h_Q.clone().detach().cpu(), z_t_Q.clone().detach().cpu()

                z_h,z_t = self.manual_single_update(z_h, z_t, idx_h,alpha)#,early)
                loss_l = (z_h.to(self.device)-z_h_Q).pow(2).mean()+ (z_t.to(self.device)-z_t_Q).pow(2).mean()

                loss = loss_l
                optim.zero_grad()
                loss.backward()
                optim.step()
            scheduler.step()
            if (((epochs+1) % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS)):
                z0 = self.vis(show=self.show,title=F"epoch{epochs+1}/{self.EPOCHS}",
                        path=None,s=self.vis_s, rainbow=self.rainbow,epochs = epochs+1)
                if self.save_Z:
                    self.Z_list.update({epochs:copy.copy(z0)})

        if self.vis_dir is not None:
            path = join(self.vis_dir, F"z_list.dat")
            torch.save(self.Z_list, path)

    def eval(self,Y,nns=[1, 2, 3, 4, 5, 10, 30]):
        Z_np = self.Z.clone().detach().numpy()
        pairwise_Z = squareform(pdist(Z_np))

        cls = knn_clf_seq(40, pairwise_Z, Y)
        results = {}
        for K in nns:
            results.update({K:cls(K)})
        return results


    def _fit_particle_only(self):
        self.Z_list.update({0:copy.copy(self.Z)})

        for epochs in range(self.EPOCHS):
            alpha = self.initial_lr - (self.initial_lr-self.end_lr) * (float(epochs) / float(self.EPOCHS))#mannual step size.

            if (epochs>5) and (epochs % 50 == 0):
                if (self.initial_sigma !=self.end_sigma):
                    sig = self.initial_sigma - (self.initial_sigma-self.end_sigma) * (float(epochs) / float(self.EPOCHS)) #**(2) #m
                    self.P_update(sig = sig)

            #early = (epochs < self.EPOCHS*0.3)
            for ii, pack in enumerate(self.g_loader):
                idx_h_dum, idx_h, idx_t_dum, idx_t = pack
                z_h, z_t = self.Z[idx_h.long()], self.Z[idx_t.long()]
                z_h, z_t = self.manual_single_update(z_h, z_t, idx_h,alpha)#,early)
                self.Z[idx_h.long()], self.Z[idx_t.long()] = z_h.float(),z_t.float()
            if (((epochs+1) % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS)):
                z0 = self.vis(show=self.show,title=F"epoch{epochs+1}/{self.EPOCHS}",
                        path=None,s=self.vis_s, rainbow=self.rainbow,epochs = epochs+1)
                if self.save_Z:
                    self.Z_list.update({epochs:copy.copy(z0)})

        if self.vis_dir is not None:
            path = join(self.vis_dir, F"z_list.dat")
            torch.save(self.Z_list, path)
