import torch
import numpy as np
from molecular_mpns.data import AlanineDipeptideGraph
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

class PhysicsVAE(torch.nn.Module):
    
    def __init__(self, xdim, zdim, encoder, decoder, U, beta = 1, graph = False, adjacency = None):
        super(PhysicsVAE, self).__init__()
        
        self.xdim = xdim
        self.zdim = zdim
        self.encoder = encoder
        self.decoder = decoder
        self.U = U
        self.beta = beta
        self.loss = {'U_loss': [], 'r_loss': [], 'H_loss': [], 'total_loss': []}
        self.converged = False
        self.improvements = []
        self.logZbeta = None
        self.c_max = 1
        self.dbeta_max = 1e-3
        self.graph = graph
        self.natoms = 5
        self.register_buffer('adjacency', adjacency)
        
    def reparameterize(self, mu_theta,logvar_theta):
        
        batch_size, device = mu_theta.shape[0], mu_theta.device
        
        sigma = torch.exp(0.5*logvar_theta) if self.graph else torch.exp(0.5*logvar_theta).repeat(batch_size, 1)
        epsilon = torch.randn((batch_size, self.xdim)).to(device)
        
        x = mu_theta + (sigma * epsilon)
        return x
    
    def forward(self, z):
        
        mu_theta, logvar_theta = self.decoder(z)
        x = self.reparameterize(mu_theta, logvar_theta)
        xnp = x.cpu().detach().numpy()
        
        if self.graph:
            dU = np.zeros((z.shape[0], 3*self.natoms))
            for i in range(xnp.shape[0]):
                x_i = xnp[i,:].reshape((self.natoms, 3))
                g = self.U.gradient(x_i)
                dU[i,:] = g
            dU = torch.tensor(dU).to(mu_theta.device)
        else:    
            dU = torch.tensor(self.U.gradient(xnp)).to(mu_theta.device)
        
        if self.graph:
            G = [AlanineDipeptideGraph(z = torch.ones(self.natoms).to(mu_theta.device),
                                       pos = torch.tensor(mol.reshape((self.natoms, 3))).to(mu_theta.device),
                                       edge_index = self.adjacency) for mol in x]
            G_batch = Batch().from_data_list(G)
            mu_phi, logvar_phi = self.encoder(x = G_batch.pos, edge_index = G_batch.edge_index, batch = G_batch.batch)
            mu_phi, logvar_phi = global_mean_pool(mu_phi, G_batch.batch), global_mean_pool(logvar_phi, G_batch.batch)
        
        else:
            mu_phi, logvar_phi = self.encoder(x)
        
        return z, mu_theta, logvar_theta, x, dU, mu_phi, logvar_phi
    
    def backprop(self, z, mu_theta, logvar_theta, x, dU, mu_phi, logvar_phi):
        
        batch_size = mu_theta.shape[0]
        
        # U loss
        xnp = x.cpu().detach().numpy()
        if self.graph:
            U_loss = 0
            for mol in xnp:
                U_loss += (1/batch_size)*self.beta*self.U(mol)
        else:    
            U_loss = (1/batch_size)*self.beta*self.U(xnp).sum()
        
        self.loss['U_loss'].append(U_loss)
        x.backward((1/batch_size)*self.beta*dU, retain_graph = True)
        
        # likelihood loss
        logvar_phi_term = 0.5*logvar_phi.sum(axis = 1)
        
        sigsq = torch.exp(logvar_phi)
        diffsq_phi = (z - mu_phi).pow(2)
        diff_term_phi = 0.5*(diffsq_phi / sigsq).sum(axis = 1)
        
        likelihood_loss = (1 / batch_size) * (logvar_phi_term + diff_term_phi).sum()
        self.loss['r_loss'].append(likelihood_loss.item())
        likelihood_loss.backward(retain_graph = True)
        
        # entropy loss
        norm_term = - z.pow(2).sum(axis = 1)
        
        if not self.graph:
            logvar_theta = logvar_theta.repeat(batch_size, 1) 
        
        logvar_theta_term = - 0.5*logvar_theta.sum(axis = 1)
        
        sigsq_theta = torch.exp(logvar_theta)
        diffsq_theta = (x - mu_theta).pow(2)
        diff_term_theta = - 0.5*(diffsq_theta / sigsq_theta).sum(axis = 1)
        
        H_loss = (1 / batch_size)*(norm_term + logvar_theta_term + diff_term_theta).sum()
        self.loss['H_loss'].append(H_loss.item())
        H_loss.backward()
        
        self.loss['total_loss'].append(U_loss + likelihood_loss.item() + H_loss.item())
        
        return U_loss.item(), likelihood_loss.item(), H_loss.item()
    
    def estimate_logZbeta0(self, z, mu_theta, logvar_theta, x, mu_phi, logvar_phi):
        
        batch_size = z.shape[0]
        
        # compute U component of log weights
        xnp = x.cpu().detach().numpy()
        if self.graph:
            U = np.zeros(batch_size)
            for i in range(xnp.shape[0]):
                U += self.U(xnp[i,:])
        
        else:        
            U = self.U(xnp)
        
        # compute log r component of log weights
        with torch.no_grad(): 
            logvar_phi_term = 0.5*logvar_phi.sum(axis = 1)
            sigsq = torch.exp(logvar_phi)
            diffsq_phi = (z - mu_phi).pow(2)
            diff_term_phi = 0.5*(diffsq_phi / sigsq).sum(axis = 1)
        logr_phi = (diff_term_phi + logvar_phi_term).cpu().numpy()
        
        # compute entropy term of log weights
        with torch.no_grad():
            norm_term = - z.pow(2).sum(axis = 1)
            if not self.graph:
                logvar_theta = logvar_theta.repeat(batch_size, 1)
            logvar_theta_term = - 0.5*logvar_theta.sum(axis = 1)
            sigsq_theta = torch.exp(logvar_theta)
            diffsq_theta = (x - mu_theta).pow(2)
            diff_term_theta = - 0.5*(diffsq_theta / sigsq_theta).sum(axis = 1)
        logq_theta = (norm_term + logvar_theta_term + diff_term_theta).cpu().numpy()
        
        # compute logZbeta0
        logw = self.beta*U + logr_phi - logq_theta
        c = logw.max()
        logw -= c
        w = np.exp(logw)
        logZbeta0 = -np.log(batch_size) + np.log(w.sum()) + c
        
        return logZbeta0
    
    def estimate_logZdiff(self, z, mu_theta, logvar_theta, x, mu_phi, logvar_phi, dbeta):
        
        batch_size = z.shape[0]
        
        # compute U component of log weights
        xnp = x.cpu().detach().numpy()
        if self.graph:
            U = np.zeros(batch_size)
            for i in range(xnp.shape[0]):
                U += self.U(xnp[i,:])
        else:
            U = self.U(xnp)
        
        # compute log r component of log weights
        with torch.no_grad(): 
            logvar_phi_term = 0.5*logvar_phi.sum(axis = 1)
            sigsq = torch.exp(logvar_phi)
            diffsq_phi = (z - mu_phi).pow(2)
            diff_term_phi = 0.5*(diffsq_phi / sigsq).sum(axis = 1)
        logr_phi = (diff_term_phi + logvar_phi_term).cpu().numpy()
        
        # compute entropy term of log weights
        with torch.no_grad():
            norm_term = - z.pow(2).sum(axis = 1)
            if not self.graph:
                logvar_theta = logvar_theta.repeat(batch_size, 1)
            logvar_theta_term = - 0.5*logvar_theta.sum(axis = 1)
            sigsq_theta = torch.exp(logvar_theta)
            diffsq_theta = (x - mu_theta).pow(2)
            diff_term_theta = - 0.5*(diffsq_theta / sigsq_theta).sum(axis = 1)
        logq_theta = (norm_term + logvar_theta_term + diff_term_theta).cpu().numpy()
        
        # compute W
        logw = self.beta*U + logr_phi - logq_theta
        c = logw.max()
        logw -= c
        w = np.exp(logw)
        W = w / w.sum()
        
        # compute log Z diff
        values = np.exp(-dbeta * U)
        weighted_sum = (values*W).sum()
        
        return np.log(weighted_sum)
        
    
    def record_improvement(self, metric):
        current_value, previous_value = metric[-1], metric[-2]
        diff = previous_value - current_value
        relative_change = diff / np.abs(previous_value)
        self.improvements.append(relative_change)
        
    def check_convergence(self, memory = 10, tol = 0.01):
        past_improvements = self.improvements[-memory:]
        self.converged = all([np.abs(imp) <= tol for imp in past_improvements])
        return self.converged
    
    def increase_beta(self, beta_target, z, mu_theta, logvar_theta, x, mu_phi, logvar_phi, U_loss, r_loss, H_loss):
        
        c_k = self.c_max + 1
        f_s = 1
        
        while (c_k > self.c_max) and (self.beta < beta_target):
            dbeta = f_s*self.dbeta_max
            logZdiff = self.estimate_logZdiff(z, mu_theta, logvar_theta, x, mu_phi, logvar_phi, dbeta)
            c_k = (logZdiff + dbeta*U_loss) / (self.logZbeta + self.beta*U_loss + r_loss + H_loss)
            f_s *= 0.6
        
        self.beta += dbeta
        self.logZbeta += logZdiff
            
        return self.beta

class DWEncoder(torch.nn.Module):
    
    def __init__(self, xdim = 2, hdim = 100, zdim = 1):
        super(DWEncoder, self).__init__()
        
        self.l1 = torch.nn.Linear(xdim, hdim)
        self.l2 = torch.nn.Linear(hdim, hdim)
        self.l3 = torch.nn.Linear(hdim, hdim)
        
        self.mu_phi = torch.nn.Linear(hdim, zdim)
        self.logvar_phi = torch.nn.Linear(hdim, zdim)
        
    def forward(self, x):
        h = torch.nn.functional.selu(self.l1(x))
        h = torch.nn.functional.selu(self.l2(h))
        h = torch.tanh(self.l3(h))
        
        mu_phi = self.mu_phi(h)
        logvar_phi = self.logvar_phi(h)
        
        return mu_phi, logvar_phi
    
class DWDecoder(torch.nn.Module):
    
    def __init__(self, zdim = 1, hdim = 100, xdim = 2):
        super(DWDecoder, self).__init__()
        
        self.logvar_theta= torch.nn.Parameter(torch.zeros(xdim), requires_grad = True)
    
        self.l1 = torch.nn.Linear(zdim, hdim)
        self.l2 = torch.nn.Linear(hdim, hdim)
        
        self.mu_theta = torch.nn.Linear(hdim, xdim)
        
    def forward(self, z):
        
        h = torch.tanh(self.l1(z))
        h = torch.tanh(self.l2(h))
        
        mu_theta = self.mu_theta(h)
        
        return mu_theta, self.logvar_theta