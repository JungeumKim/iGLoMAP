import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import torch
from iglo.manifold.neighbor_dataset import Neighbor_dataset
from iglo.manifold._util_manifold import torch_pairwise_distances,random_nb_sparse,rand_seed
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_
import torch.nn as nn
from iglo.evals.eval_tools import knn_clf_seq
from scipy.spatial.distance import pdist, squareform
# model: to initialize
import iglo._networks.network_conv_grey as ncg
import iglo._networks.network_toy as nt
from os.path import join
#hyper params :
MACHINE_EPSILON = np.finfo(np.double).eps

class  Glomap():
    def __init__(self,
                 n_neighbors = 5,
                 d_thresh = 10,
                 n_neg_rate = 100,
                 glob_n_neg_rate=5,
                 ee = 0.1,
                 a = 1.57694,
                 b = 0.8951,
                 glob_a = 1.57694,
                 glob_b = 0.8951,
                 initial_lr = 1,
                 end_lr = 0,
                 batch_size = 100,
                 fmax = True,
                 z_dim=2,
                 EPOCHS = 500,
                 plot_freq = 20,
                 optimizer=None,
                 seed=1234,
                 show=True,
                 vis_dir = None,
                vis_s = 1,
                prob_neg = False,
                 glob_prob_neg=True,
                _lambda=0,
                 clamp=4,
                 inductive=False,
                 device="cuda",
                 lr_Q = 0.01,
                 conv=True,
                 Q = None):

        ''' ARGUMENTS:
        optimizer: if None, manual gradient steps are used. else (e.g., sgd), then the SGD torch implementation used.
            But anyway because any momentum based method will not be used, either one should have no difference.
        n_neighbors: number of neighbors used to construct the graph for global geodesic distance estimates
        n_neg_rate: for one positive sample, how many neg samples will be used
        ee: the weight of the loss for the negative part
        device, lr_Q, conv: only used when inductive is True.
        '''
        self.n_neg_rate =  min(n_neg_rate,batch_size)
        self.glob_n_neg_rate = min(glob_n_neg_rate, batch_size)
        self.d_thresh = d_thresh
        self.n_neighbors = n_neighbors
        self.ee = ee
        self.a = a
        self.b = b
        self.initial_lr = initial_lr
        self.end_lr=end_lr
        self.batch_size = batch_size
        self.fmax = fmax
        self.z_dim = z_dim
        self.EPOCHS = EPOCHS
        self.plot_freq = plot_freq
        self.optimizer = optimizer
        self.seed = seed
        self.show = show
        self.vis_dir = vis_dir
        self.vis_s = vis_s
        self.prob_neg = prob_neg
        self._lambda = _lambda
        self.glob_prob_neg = glob_prob_neg
        self.glob_b = glob_b
        self.glob_a =glob_a
        self.clamp= clamp
        self.inductive = inductive
        if inductive:
            self.device = device
        else:
            self.device = "cpu"
        self.lr_Q = lr_Q
        self.conv= conv
        self.Q = Q

    def sparse_P_update(self,new_P_data):
        self.sparse_P = new_P_data
        v_i_dot = self.sparse_P.sum(axis=1)
        v_i_dot = torch.from_numpy(v_i_dot)
        vm = v_i_dot.max()
        v_i_dot /= vm
        self.v_i_dot = v_i_dot

        def random_idx_generator(idx):
            return random_nb_sparse(self.sparse_P, idx)

        indices = np.array(range(self.sparse_P.shape[0])).reshape(-1, 1)
        indx_dataset_g = Neighbor_dataset(indices,
                                          neighbor_rule=random_idx_generator,
                                          return_idx=True)

        self.g_loader = DataLoader(indx_dataset_g,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   num_workers=2)

    def _fit_prepare(self, X,Y, precalc_graph=None,network_init = True):
        if precalc_graph is None:
            #g_dist = iglo_graph(X, self.n_neighbors, self.fmax)
            raise NotImplementedError
        else:
            g_dist = precalc_graph
        if self._lambda>0: #global
            gg_dist = np.copy(g_dist)
            #gg_dist[gg_dist == 0] = float("inf")
            mu = np.exp(-gg_dist)
            np.fill_diagonal(mu, 0)
            self.mu = torch.tensor(mu)

        if self._lambda<1: #local
            g_dist[g_dist > self.d_thresh] = float("inf")

            #g_dist[g_dist == 0] = float("inf")
            P = np.exp(-g_dist)
            np.fill_diagonal(P, 0)
            del g_dist
            self.sparse_P = sparse.csr_matrix(P)
            del P

            v_i_dot = self.sparse_P.sum(axis=1)
            v_i_dot = torch.from_numpy(v_i_dot)
            vm = v_i_dot.max()
            v_i_dot /= vm
            self.v_i_dot = v_i_dot
        self.learning_ee = self.ee

        if self._lambda<1:
            def random_idx_generator(idx):
                return random_nb_sparse(self.sparse_P, idx)
        else: #when only fully global, we don't need neighbor sampling
            def random_idx_generator(idx):
                return idx
        indices = np.array(range(X.shape[0])).reshape(-1, 1)
        indx_dataset_g = Neighbor_dataset(indices,
                                   neighbor_rule=random_idx_generator,
                                   return_idx=True)

        self.g_loader = DataLoader(indx_dataset_g,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=2)
        #indx_dataset_c = Np_dataset(indices)
        #self.c_loader = DataLoader(indx_dataset_c,
        #                           batch_size=self.batch_size,
        #                             shuffle=True,
        #                            num_workers=2)

        if network_init or self.inductive:
            rand_seed(self.seed)
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

            if self.inductive:
                self.Q = Q.to(self.device)
        else:
            generator = np.random.RandomState(self.seed)
            self.Z = torch.from_numpy(
                            generator.normal(loc=0.0, scale=.2,
                             size=(X.shape[0], self.z_dim)
                            )
            ).float()
        self.Y = Y
        if self.inductive:
            self.X = torch.from_numpy(X).float()
        if self.prob_neg and (self._lambda<1) :
            self.nu = torch.tensor(self.sparse_P.todense())

    def fit_transform(self, X, Y=None,precalc_graph=None, network_init=True, eval=True):
        self._fit_prepare(X, Y, precalc_graph, network_init)
        print("The learning is prepared")
        
        if self.optimizer is None:
            print("mode:manual gradient update")
            if self.inductive:
                self._fit_particle()

            else:
                self._fit_manual()
        else:
            if self.inductive:
                assert self._lambda == 0, "implemented for only neighbor sampler"
                self._fit_optim()
            else:
                print("mode:torch.sgd gradient update")
                self._fit_SGD()
        #set_trace()
        if (eval) and (not  isinstance(Y, type(None))):
            self.eval_result = self.eval(Y)
            print("eval for knns:",self.eval_result)

        return self.Z
    def vis(self,Y=None,axis=None, s=1, show=False,title=None,path=None,rainbow=False):

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

        if self.inductive:
            Z0 = []
            with torch.no_grad():
                for i in range(int(self.X.shape[0]/self.batch_size)):
                    z = self.Q(self.X[i*self.batch_size: (i+1)*self.batch_size].to(self.device))
                    Z0.append(z.cpu())
                if (self.X.shape[0]%self.batch_size)>0:
                    i=i+1
                    z = self.Q(self.X[i*self.batch_size: ].to(self.device))
                    Z0.append(z.cpu())
            self.Z = torch.cat(Z0)
            Z0 = self.Z.numpy()

        else:
            Z0 = self.Z.clone().detach().numpy()
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
        plt.close()

    def manual_single_negative_grad(self,z_h,a,b,):
        z_dist_square = torch_pairwise_distances(z_h, dim=2).pow(2)
        deno = (0.001 + z_dist_square) * (a * z_dist_square.pow(b) + 1)

        grad_coeff = 2.0 * b
        grad_coeff = grad_coeff / deno
        # important : torch.pairwise_distances gives other - z_h. Therefore, should multiply by -1.
        diff = -torch_pairwise_distances(z_h, dim=2, reduction=None)
        neg_step = grad_coeff.unsqueeze(2) * diff
        neg_step = neg_step.clamp(-self.clamp, self.clamp)
        return neg_step
    def manual_single_update(self, z_h, z_t, idx_h,alpha):
        ## negative step
        neg_grad = self.manual_single_negative_grad(z_h,self.a, self.b)
        if self.prob_neg:
            #one_minus_mu = 1-self.sparse_P[idx_h, :][:, idx_h].todense()
            #neg_step = torch.tensor(one_minus_mu).unsqueeze(2) * neg_step
            one_minus_mu = 1-self.nu[idx_h, :][:, idx_h]
            neg_step = one_minus_mu.unsqueeze(2) * neg_grad
        else:
            neg_step = neg_grad
        neg_step = neg_step[:, :self.n_neg_rate, :].sum(dim=1)

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

    def manual_global_single_update(self, z_h, idx_h, alpha):
        #note: z_t will not be updated at all. That's the rule.

        # -- neg_step
        neg_grad = self.manual_single_negative_grad(z_h, self.glob_a, self.glob_b)
        if self.glob_prob_neg:
            #one_minus_mu = 1-self.sparse_P[idx_h, :][:, idx_h].todense()
            #neg_step = torch.tensor(one_minus_mu).unsqueeze(2) * neg_step
            one_minus_mu = 1-self.mu[idx_h, :][:, idx_h]
            neg_step = one_minus_mu.unsqueeze(2) * neg_grad
        else:
            neg_step = neg_grad
        neg_step = neg_step[:, :self.glob_n_neg_rate, :].sum(dim=1)
        updated_z_h = z_h + alpha * self.learning_ee * neg_step

        # -- pos_step
        z_dist_square = torch_pairwise_distances(updated_z_h, dim=2).pow(2)
        if self.glob_b <= 1:
            z_dist_square.fill_diagonal_(np.inf) 
        grad_coeff = -2.0 * self.glob_a * self.glob_b * z_dist_square.pow(self.glob_b - 1.0)
        if self.glob_b>1:
            z_dist_square.fill_diagonal_(np.inf)        
        grad_coeff /= self.glob_a * z_dist_square.pow(self.glob_b) + 1.0
        mu = self.mu[idx_h, :][:, idx_h]
        grad_coeff *= mu

        diff = -torch_pairwise_distances(updated_z_h, dim=2, reduction=None)
        pos_step = grad_coeff.unsqueeze(2) * diff
        pos_step = pos_step.clamp(-self.clamp, self.clamp)
        pos_step = pos_step.sum(dim=1)

        next_z_h = updated_z_h + alpha * pos_step
        return next_z_h

    def _fit_SGD(self):
        Z = self.Z.clone().float()
        Z = nn.parameter.Parameter(data=Z, requires_grad=True)

        # Adam or momentums are not ideal. Because in each Z update, I will partially update it.
        # Then the gradients are mostly zero, which reduces the varaince than originally it should be.
        optim_pos = torch.optim.SGD([Z], lr=self.initial_lr)
        optim_neg = torch.optim.SGD([Z], lr=self.initial_lr)

        scheduler_pos = torch.optim.lr_scheduler.ExponentialLR(optim_pos, gamma=0.99)
        scheduler_neg = torch.optim.lr_scheduler.ExponentialLR(optim_neg, gamma=0.99)

        cri_neg = nn.BCELoss(reduction='none')
        cri_pos = nn.BCELoss(reduction='none')

        for epochs in range(self.EPOCHS):
            for ii, pack in enumerate(self.g_loader):

                idx_h_dum, idx_h, idx_t_dum, idx_t = pack

                z_h, z_t = Z[idx_h.long()], Z[idx_t.long()]

                ## negative step
                z_dist_square = torch_pairwise_distances(z_h, dim=2).pow(2 * self.b)
                z_dist_square = z_dist_square[:, :self.n_neg_rate]
                prob = 1.0 / (1.0 + self.a * z_dist_square)
                prob.fill_diagonal_(0)
                for j in range(self.n_neg_rate):
                    loss_neg = cri_neg(prob[:, j].view(-1), torch.zeros(prob.shape[0]))
                    if self.prob_neg:
                        '''
                            only when pob_neg, we weight each loss 
                            element by (1-mu_ij) before summing up.
                        '''
                        # one_minus_mu = 1 - self.sparse_P[idx_h, j].todense()
                        # loss_neg = torch.tensor(one_minus_mu).view(-1) * loss_neg
                        one_minus_mu = 1 - self.nu[idx_h, j]
                        loss_neg = one_minus_mu.view(-1) * loss_neg
                    loss_neg = loss_neg.sum()
                    optim_neg.zero_grad()
                    loss_neg.backward(retain_graph=True)

                    clip_grad_value_([Z], self.clamp)
                    Z.grad.data *= self.learning_ee
                    # print(Z.grad.max(), prob.max())
                    optim_neg.step()
                del loss_neg

                ## positive step
                updated_z_h = Z[idx_h.long()]
                znb_dist_square = (updated_z_h - z_t).norm(dim=1).pow(2 * self.b)
                prob = 1.0 / (1.0 + self.a * znb_dist_square)
                loss_pos = cri_pos(prob.view(-1), torch.ones(prob.shape[0]))
                loss_pos *= self.v_i_dot[idx_h].clone().detach().view(-1)
                loss_pos = loss_pos.sum()
                optim_pos.zero_grad()
                loss_pos.backward()
                clip_grad_value_([Z], self.clamp)
                optim_pos.step()

            scheduler_pos.step()
            scheduler_neg.step()

            if ((epochs % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS)):
                self.Z = Z.clone().detach().to("cpu")

                path = join(self.vis_dir, F"epoch{epochs + 1}")
                self.vis(show=self.show, title=F"epoch{epochs + 1}/{self.EPOCHS}", path=path, s=self.vis_s)

    def _fit_manual(self):
        for epochs in range(self.EPOCHS):
            alpha = self.initial_lr - (self.initial_lr-self.end_lr) * (float(epochs) / float(self.EPOCHS))
            #early = (epochs < self.EPOCHS*0.5)
            #alpha = self.initial_lr * (1.0 - (float(epochs) / float(self.EPOCHS))) #mannual step size.
            for ii, pack in enumerate(self.g_loader):

                idx_h_dum, idx_h, idx_t_dum, idx_t = pack

                z_h, z_t = self.Z[idx_h.long()], self.Z[idx_t.long()]
                if self._lambda<1: #local step
                    z_h,z_t = self.manual_single_update(z_h, z_t, idx_h,alpha*(1-self._lambda))#, early)
                    self.Z[idx_h.long()] = z_h
                    self.Z[idx_t.long()] = z_t

                if self._lambda>0: #global step
                    z_h = self.manual_global_single_update(z_h,idx_h, alpha*self._lambda)
                    self.Z[idx_h.long()] = z_h

            if ((epochs % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS)):
                if self.vis_dir is None:
                    path=None
                else:
                    path = join(self.vis_dir, F"epoch{epochs + 1}")
                self.vis(show=self.show,title=F"epoch{epochs+1}/{self.EPOCHS}", path=path,s=self.vis_s)

    def _current_loss(self):
        cri_neg = nn.BCELoss(reduction='none')
        cri_pos = nn.BCELoss(reduction='none')
        loss = 0
        with torch.no_grad():
            for ii, pack in enumerate(self.g_loader):
                idx_h_dum, idx_h, idx_t_dum, idx_t = pack
                z_h, z_t = self.Z[idx_h.long()], self.Z[idx_t.long()]

                ## negative step
                z_dist_square = torch_pairwise_distances(z_h, dim=2).pow(2 * self.b)
                prob = 1.0 / (1.0 + self.a * z_dist_square)
                prob.fill_diagonal_(0)
                loss_neg = cri_neg(prob.view(-1), torch.zeros(prob.view(-1).shape[0]) ).sum() * self.learning_ee

                znb_dist_square = (z_h - z_t).norm(dim=1).pow(2 * self.b)
                prob = 1.0 / (1.0 + self.a * znb_dist_square)
                loss_pos = cri_pos(prob.view(-1), torch.ones(prob.shape[0]))
                loss_pos *= self.v_i_dot[idx_h].view(-1)
                loss_pos = loss_pos.sum()

                loss += loss_neg+loss_pos
        return loss

    def _fit_particle(self):
        optim = torch.optim.Adam(self.Q.parameters(), lr=self.lr_Q)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
        for epochs in range(self.EPOCHS):
            alpha = self.initial_lr - (self.initial_lr-self.end_lr) * (float(epochs) / float(self.EPOCHS)) #mannual step size.
            #early = (epochs < self.EPOCHS*0.3)
            for ii, pack in enumerate(self.g_loader):

                idx_h_dum, idx_h, idx_t_dum, idx_t = pack

                x_h, x_t = self.X[idx_h.long()].to(self.device), self.X[idx_t.long()].to(self.device)
                z_h_Q, z_t_Q = self.Q(x_h), self.Q(x_t)
                z_h, z_t = z_h_Q.clone().detach().cpu(), z_t_Q.clone().detach().cpu()

                if self._lambda<1: #local step
                    z_h,z_t = self.manual_single_update(z_h, z_t, idx_h,alpha*(1-self._lambda))#,early)
                    loss_l = (z_h.to(self.device)-z_h_Q).pow(2).mean()+ (z_t.to(self.device)-z_t_Q).pow(2).mean()
                else:
                    loss_l = torch.zeros(1).to(self.device)

                if self._lambda>0: #global step
                    z_hg = self.manual_global_single_update(z_h,idx_h, alpha*self._lambda)
                    loss_g = (z_hg.to(self.device)-z_h_Q).pow(2).mean()
                else:
                    loss_g = torch.zeros(1).to(self.device)
                loss = loss_l + loss_g
                optim.zero_grad()
                loss.backward()
                optim.step()
            scheduler.step()
            if ((epochs % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS)):
                if self.vis_dir is None:
                    path=None
                else:
                    path = join(self.vis_dir, F"epoch{epochs + 1}")
                self.vis(show=self.show,title=F"epoch{epochs+1}/{self.EPOCHS}", 
                        path=path,s=self.vis_s)

    def _fit_optim(self):
        optim_pos = torch.optim.Adam(self.Q.parameters(), lr=self.lr_Q)
        optim_neg = torch.optim.Adam(self.Q.parameters(), lr=self.lr_Q)
        scheduler_pos = torch.optim.lr_scheduler.ExponentialLR(optim_pos, gamma=0.95)
        scheduler_neg = torch.optim.lr_scheduler.ExponentialLR(optim_neg, gamma=0.95)
        for epochs in range(self.EPOCHS):
            #alpha = self.initial_lr - (self.initial_lr-self.end_lr) * (float(epochs) / float(self.EPOCHS)) #mannual step size.
            for ii, pack in enumerate(self.g_loader):

                idx_h_dum, idx_h, idx_t_dum, idx_t = pack

                x_h, x_t = self.X[idx_h.long()].to(self.device), self.X[idx_t.long()].to(self.device)
                z_h =  self.Q(x_t)

                ## negative step
                z_dist_square = torch_pairwise_distances(z_h, dim=2).pow(2 * self.b)
                z_dist_square = z_dist_square[:, :self.n_neg_rate]
                prob = 1.0 / (1.0 + self.a * z_dist_square)
                loss_neg = -(1-prob+MACHINE_EPSILON).log()
                loss_neg *= (1-torch.eye(loss_neg.shape[0],device=loss_neg.device))                
                loss_neg = loss_neg.sum()* self.learning_ee
                optim_neg.zero_grad()
                loss_neg.backward()
                optim_neg.step()

                ## positive step
                updated_z_h, z_t = self.Q(x_h), self.Q(x_t)
                znb_dist_square = (updated_z_h - z_t).norm(dim=1).pow(2 * self.b)
                prob = 1.0 / (1.0 + self.a * znb_dist_square)
                loss_pos = -(prob+MACHINE_EPSILON).log()
                loss_pos *= self.v_i_dot[idx_h].clone().detach().view(-1).to(loss_pos.device)
                loss_pos = loss_pos.sum()
                optim_pos.zero_grad()
                loss_pos.backward()
                optim_pos.step()

            scheduler_pos.step()
            scheduler_neg.step()

            if ((epochs % self.plot_freq == 0) or (epochs + 1 == self.EPOCHS)):
                if self.vis_dir is None:
                    path=None
                else:
                    path = join(self.vis_dir, F"epoch{epochs + 1}")
                self.vis(show=self.show,title=F"epoch{epochs+1}/{self.EPOCHS}",
                        path=path,s=self.vis_s)

    def eval(self,Y,nns=[1, 2, 3, 4, 5, 10, 30]):
        Z_np = self.Z.clone().detach().numpy()
        pairwise_Z = squareform(pdist(Z_np))

        cls = knn_clf_seq(40, pairwise_Z, Y)
        results = {}
        for K in nns:
            results.update({K:cls(K)})
        return results
