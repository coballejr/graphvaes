#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 15:40:33 2021

@author: chris
"""

import numpy as np
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import radius_graph
from torch_geometric.nn import global_mean_pool

class Encoder(torch.nn.Module):
    
    def __init__(self,in_channels, hidden_channels, out_channels, num_hidden_layers, act = torch.nn.functional.silu):
        super(Encoder, self).__init__()
        
        self.act = act
        self.gc_layers = torch.nn.ModuleList([GCNConv(in_channels = in_channels, out_channels = hidden_channels)])
        
        for l in range(num_hidden_layers):
            gc_layer = GCNConv(in_channels = hidden_channels, out_channels = hidden_channels)
            self.gc_layers.append(gc_layer)
        
        self.mu = GCNConv(in_channels = hidden_channels, out_channels = out_channels)
        self.logvar = GCNConv(in_channels = hidden_channels, out_channels = out_channels)
        
    def forward(self, x, edge_index = None, cutoff = None, batch = None):
        
        if edge_index is None:
            if cutoff:
                edge_index = radius_graph(x, r = cutoff, batch = batch)
            else:
                raise ValueError('No adjacency matrix or cutoff parameter specified.')
        
        for l, gc_layer in enumerate(self.gc_layers):
            if l == 0:
                x = torch.nn.functional.selu(gc_layer(x, edge_index))
            elif l == (len(self.gc_layers) - 1):
                x = torch.log(torch.nn.functional.sigmoid(gc_layer(x, edge_index)))
            else:
                x = self.act(gc_layer(x, edge_index))
        
        mu_z = self.mu(x, edge_index)
        logvar_z = self.logvar(x, edge_index)
        
        return mu_z, logvar_z

class Decoder(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_hidden_layers, act = torch.tanh, var_learnable = True):
        super(Decoder,self).__init__()
        
        self.act = act
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_channels, hidden_channels)])
        
        for l in range(num_hidden_layers):
            lin = torch.nn.Linear(hidden_channels, hidden_channels)
            self.lins.append(lin)
        
        self.mu_x = torch.nn.Linear(hidden_channels, out_channels)
        self.logvar_x = torch.nn.Parameter(torch.zeros(out_channels), requires_grad = True) if var_learnable else torch.ones(out_channels)
        
    def forward(self, z):
        
        for l, lin in enumerate(self.lins):
            if ((l == 0) or (l == (len(self.lins) - 1))):
                z = torch.tanh(lin(z))
            else:
                z = self.act(lin(z))
        
        mu_x = self.mu_x(z)
        logvar_x = self.logvar_x.repeat(mu_x.shape[0],1)
        
        return mu_x, logvar_x
    
class ResEncoder(torch.nn.Module):
    
    def __init__(self,in_channels, hidden_channels, out_channels, num_hidden_layers, act = torch.nn.functional.silu):
        super(ResEncoder, self).__init__()
        
        self.act = act
        self.gc_layers = torch.nn.ModuleList([GCNConv(in_channels = in_channels, out_channels = hidden_channels)])
        
        for l in range(num_hidden_layers):
            gc_layer = GCNConv(in_channels = hidden_channels, out_channels = hidden_channels)
            self.gc_layers.append(gc_layer)
        
        self.mu = GCNConv(in_channels = hidden_channels, out_channels = out_channels)
        self.logvar = GCNConv(in_channels = hidden_channels, out_channels = out_channels)
        
    def forward(self, x, edge_index = None, cutoff = None, batch = None):
        
        if edge_index is None:
            if cutoff:
                edge_index = radius_graph(x, r = cutoff, batch = batch)
            else:
                raise ValueError('No adjacency matrix or cutoff parameter specified.')
        
        for l, gc_layer in enumerate(self.gc_layers):
            if l == 0:
                x = torch.nn.functional.selu(gc_layer(x, edge_index)) 
            elif l == (len(self.gc_layers) - 1):
                x = torch.log(torch.nn.functional.sigmoid(gc_layer(x, edge_index))) + x
            else:
                x = self.act(gc_layer(x, edge_index)) + x
        
        mu_z = self.mu(x, edge_index)
        logvar_z = self.logvar(x, edge_index)
        
        return mu_z, logvar_z

class ResDecoder(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_hidden_layers, act = torch.tanh, var_learnable = True):
        super(ResDecoder,self).__init__()
        
        self.act = act
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_channels, hidden_channels)])
        
        for l in range(num_hidden_layers):
            lin = torch.nn.Linear(hidden_channels, hidden_channels)
            self.lins.append(lin)
        
        self.mu_x = torch.nn.Linear(hidden_channels, out_channels)
        self.logvar_x = torch.nn.Parameter(torch.zeros(out_channels), requires_grad = True) if var_learnable else torch.ones(out_channels)
        
    def forward(self, z):
        
        for l, lin in enumerate(self.lins):
            if ((l == 0) or (l == (len(self.lins) - 1))):
                z = torch.tanh(lin(z))
            else:
                z = self.act(lin(z)) + z
        
        mu_x = self.mu_x(z)
        logvar_x = self.logvar_x.repeat(mu_x.shape[0],1)
        
        return mu_x, logvar_x
    
class GraphVAE(torch.nn.Module):
    
    def __init__(self, x_dim, enc_hidden_channels, z_dim, num_enc_hidden_layers, dec_hidden_channels, num_dec_hidden_layers, enc_act, dec_act ,n_atoms = 22, cutoff = 0.5, adjacency = None, output_dists = True):
        super(GraphVAE,self).__init__()
        self.x_dim = x_dim
        self.n_atoms = n_atoms
        self.z_dim = z_dim
        self.cutoff = cutoff
        self.register_buffer('adjacency', adjacency)
        self.encoder = Encoder(in_channels = x_dim, hidden_channels = enc_hidden_channels, out_channels = z_dim,
                               num_hidden_layers = num_enc_hidden_layers, act = enc_act)
        dec_out_channels = int(x_dim*n_atoms) if not output_dists else int(n_atoms*(n_atoms-1) / 2)
        self.decoder = Decoder(in_channels = z_dim, hidden_channels = dec_hidden_channels, out_channels = dec_out_channels, num_hidden_layers = num_dec_hidden_layers, act = dec_act)
        
    def reparameterize(self ,mu_z, logvar_z):
        sigma = torch.exp(0.5*logvar_z)
        eps = torch.randn_like(sigma)
        
        z = mu_z + (sigma*eps)
        
        return z
    
    def forward(self, x, edge_index = None, batch = None):
        mu_z, logvar_z = self.encoder(x, edge_index = edge_index, cutoff= self.cutoff, batch = batch)
        mu_z = global_mean_pool(mu_z, batch)
        logvar_z = global_mean_pool(logvar_z, batch)
        
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decoder(z)
        
        return mu_z, logvar_z, mu_x, logvar_x
    
    def mwg_sample(self, N):
        dev = self.encoder.mu.weight.device
        samples = np.zeros((N, self.n_atoms * self.x_dim))
        x, z = torch.randn((self.n_atoms, self.x_dim)).to(dev), torch.randn((1,self.z_dim)).to(dev)
        
        for n in range(N):
            with torch.no_grad():
                mu_z, logvar_z = self.encoder(x, edge_index = self.adjacency, cutoff = self.cutoff)
                zprop = self.reparameterize(mu_z, logvar_z)
            
            # compute accepatance ratio
            with torch.no_grad():
                mu_x, logvar_x = self.decoder(z)
                mu_xprop, logvar_xprop = self.decoder(zprop)
            
            p_x_given_z = torch.exp(-0.5*(((x.flatten() - mu_x)**2) / torch.exp(logvar_x)).sum())
            p_x_given_zprop = torch.exp(-0.5*(((x.flatten() - mu_xprop)**2) / torch.exp(logvar_xprop)).sum())
            
            p_z = torch.exp(-0.5*(z**2).sum())
            p_zprop = torch.exp(-0.5*(zprop**2).sum())
        
            q_z_given_x = torch.exp(-0.5*(((z - mu_z)**2) / torch.exp(logvar_z)).sum())
            q_zprop_given_x = torch.exp(-0.5*(((zprop - mu_z)**2) / torch.exp(logvar_z)).sum())
            
            ar = (p_x_given_zprop / p_x_given_z)*(p_zprop / p_z)*(q_z_given_x / q_zprop_given_x)
            draw = np.random.rand()
            
            z = zprop if draw < ar.item() else z
            
            with torch.no_grad():
                mu_x, logvar_x= self.decoder(z)
                x = self.reparameterize(mu_x, logvar_x)
            
            x_np = x.cpu().numpy()
            samples[n,: ] = x_np
            
            x = x.reshape((self.n_atoms, self.x_dim))
            
        return samples
    
    def ancestral_sample(self, N):
        dev = self.encoder.mu.weight.device
        z = torch.randn((N,self.z_dim)).to(dev)
        with torch.no_grad():
            mu_x, logvar_x = self.decoder(z)
            x = self.reparameterize(mu_x, logvar_x)
        
        return x.cpu().numpy()
    
    def sample_decoder_means(self,N):
        dev = self.encoder.mu.weight.device
        z = torch.randn((N,self.z_dim)).to(dev)
        with torch.no_grad():
            mu_x, _ = self.decoder(z)
        return mu_x.cpu().numpy()
    
class ResGraphVAE(torch.nn.Module):
    
    def __init__(self, x_dim, enc_hidden_channels, z_dim, num_enc_hidden_layers, dec_hidden_channels, num_dec_hidden_layers, enc_act, dec_act ,n_atoms = 22, cutoff = 0.5, adjacency = None, output_dists = True):
        super(ResGraphVAE,self).__init__()
        self.x_dim = x_dim
        self.n_atoms = n_atoms
        self.z_dim = z_dim
        self.cutoff = cutoff
        self.register_buffer('adjacency', adjacency)
        self.encoder = ResEncoder(in_channels = x_dim, hidden_channels = enc_hidden_channels, out_channels = z_dim,
                               num_hidden_layers = num_enc_hidden_layers, act = enc_act)
        dec_out_channels = int(x_dim*n_atoms) if not output_dists else int(n_atoms*(n_atoms-1) / 2)
        self.decoder = ResDecoder(in_channels = z_dim, hidden_channels = dec_hidden_channels, out_channels = dec_out_channels, num_hidden_layers = num_dec_hidden_layers, act = dec_act)
        
    def reparameterize(self ,mu_z, logvar_z):
        sigma = torch.exp(0.5*logvar_z)
        eps = torch.randn_like(sigma)
        
        z = mu_z + (sigma*eps)
        
        return z
    
    def forward(self, x, edge_index = None, batch = None):
        mu_z, logvar_z = self.encoder(x, edge_index = edge_index, cutoff= self.cutoff, batch = batch)
        mu_z = global_mean_pool(mu_z, batch)
        logvar_z = global_mean_pool(logvar_z, batch)
        
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decoder(z)
        
        return mu_z, logvar_z, mu_x, logvar_x
    
    def mwg_sample(self, N):
        dev = self.encoder.mu.weight.device
        samples = np.zeros((N, self.n_atoms * self.x_dim))
        x, z = torch.randn((self.n_atoms, self.x_dim)).to(dev), torch.randn((1,self.z_dim)).to(dev)
        
        for n in range(N):
            with torch.no_grad():
                mu_z, logvar_z = self.encoder(x, edge_index = self.adjacency, cutoff = self.cutoff)
                zprop = self.reparameterize(mu_z, logvar_z)
            
            # compute accepatance ratio
            with torch.no_grad():
                mu_x, logvar_x = self.decoder(z)
                mu_xprop, logvar_xprop = self.decoder(zprop)
            
            p_x_given_z = torch.exp(-0.5*(((x.flatten() - mu_x)**2) / torch.exp(logvar_x)).sum())
            p_x_given_zprop = torch.exp(-0.5*(((x.flatten() - mu_xprop)**2) / torch.exp(logvar_xprop)).sum())
            
            p_z = torch.exp(-0.5*(z**2).sum())
            p_zprop = torch.exp(-0.5*(zprop**2).sum())
        
            q_z_given_x = torch.exp(-0.5*(((z - mu_z)**2) / torch.exp(logvar_z)).sum())
            q_zprop_given_x = torch.exp(-0.5*(((zprop - mu_z)**2) / torch.exp(logvar_z)).sum())
            
            ar = (p_x_given_zprop / p_x_given_z)*(p_zprop / p_z)*(q_z_given_x / q_zprop_given_x)
            draw = np.random.rand()
            
            z = zprop if draw < ar.item() else z
            
            with torch.no_grad():
                mu_x, logvar_x= self.decoder(z)
                x = self.reparameterize(mu_x, logvar_x)
            
            x_np = x.cpu().numpy()
            samples[n,: ] = x_np
            
            x = x.reshape((self.n_atoms, self.x_dim))
            
        return samples
    
    def ancestral_sample(self, N):
        dev = self.encoder.mu.weight.device
        z = torch.randn((N,self.z_dim)).to(dev)
        with torch.no_grad():
            mu_x, logvar_x = self.decoder(z)
            x = self.reparameterize(mu_x, logvar_x)
        
        return x.cpu().numpy()
    
    def sample_decoder_means(self,N):
        dev = self.encoder.mu.weight.device
        z = torch.randn((N,self.z_dim)).to(dev)
        with torch.no_grad():
            mu_x, _ = self.decoder(z)
        return mu_x.cpu().numpy()
        
        
            
        
def VAEloss(data, mu_x, logvar_x ,mu_z, logvar_z, kl_weight = 1, logvar_weight = 1, tol = 1e-7):
    
    # recon loss for p(x | z)
    pointwiseMSEloss = 0.5*torch.nn.functional.mse_loss(mu_x, data,reduction = 'none')
    sigsq = torch.exp(logvar_x) + tol
    weight = 1/sigsq
    pointwiseWeightedMSEloss = pointwiseMSEloss*weight
    WeightedMSEloss = pointwiseWeightedMSEloss.sum()
    
    logvarobjective = 0.5 * logvar_x.sum() # scaling factor term for p(x|z)
    
    # KLD loss for q(z | x)
    KLD = -0.5 * torch.sum(1 + logvar_z - mu_z**2 - torch.exp(logvar_z))
    
    loss = (kl_weight*KLD + WeightedMSEloss + logvar_weight*logvarobjective)
    
    return loss, KLD, WeightedMSEloss, logvarobjective