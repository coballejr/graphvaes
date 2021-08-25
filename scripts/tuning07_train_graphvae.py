# imports
import numpy as np
import mdtraj as md
from molecular_mpns.data import AlanineDipeptideGraph, bond_adjacency_matrix
from molecular_mpns.training_utils import train_test_split
from molecular_mpns.graphvae import GraphVAE, VAEloss
from molecular_mpns.math import radius_of_gyration
from torch_geometric.data import DataLoader
import torch
import matplotlib.pyplot as plt
import os


# load training data
data_dir = '/afs/crc.nd.edu/user/c/coballe/graphvaes/data/'
peptide_file = 'dataset_500.txt'
pdb_file = 'ala2_adopted.pdb'

xyz = np.loadtxt(data_dir + peptide_file).T
topology = md.load(data_dir + pdb_file).topology

traj_xyz = xyz.reshape((xyz.shape[0],22,3))
traj = md.Trajectory(xyz = traj_xyz, topology = topology)

# create graphs
z = [atom.element.atomic_number for atom in traj.topology.atoms]
adjacency = bond_adjacency_matrix(traj.topology)
G = [AlanineDipeptideGraph(z = torch.tensor(z).long(),pos = torch.tensor(xyz), edge_index = adjacency) for xyz in traj.xyz]

# build model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params = {'x_dim': 3, 'enc_hidden_channels': 64, 'z_dim': 32, 'num_enc_hidden_layers': 4,
                 'dec_hidden_channels': 64, 'num_dec_hidden_layers': 4, 'adjacency': adjacency, 'output_dists': False,
                 'enc_act': torch.nn.functional.silu, 'dec_act': torch.tanh}
mod = GraphVAE(**params)
mod = mod.to(device)
opt = torch.optim.Adam(mod.parameters(), lr = 1e-3, weight_decay = 1e-4)

# train model
fig_dir = '/afs/crc.nd.edu/user/c/coballe/graphvaes/figs/tuning_architectures/07/'
model_dir = '/afs/crc.nd.edu/user/c/coballe/graphvaes/models/tuning_architectures/'
os.chdir(fig_dir)

epochs, batch_size, cutoff, n_atoms = 8000, 64, 0.5, 22
subset_size, train_prop  = 500, 1.0
train, test = train_test_split(data = G,subset_size = subset_size, train_prop = train_prop)

train_xyz = np.array([G.pos.cpu().numpy() for G in train])
train_traj = md.Trajectory(xyz = train_xyz, topology = traj.topology)

psi_inds = [6, 8, 14, 16]
phi_inds = [4, 6, 8, 14]
tr_dangles = md.compute_dihedrals(train_traj,[psi_inds,phi_inds])
tr_rogs = [radius_of_gyration(xyz, topology = topology) for xyz in train_xyz]

running_loss, running_kld_loss, running_mse_loss, running_logvar_loss = [], [], [], []

for ep in range(epochs):
    kl_weight = ep / epochs
    ep_loss = 0
    ep_kld_loss = 0
    ep_mse_loss = 0
    ep_logvar_loss = 0
    
    # shuffle training set
    np.random.seed(42)
    random_idx = np.random.choice(len(train),len(train), replace = False)
    G_epoch = [train[i] for i in random_idx]
    loader = DataLoader(G_epoch,batch_size = batch_size)
    
    for G_batch in loader:
        
        # forward pass
        G_batch = G_batch.to(device)
        mu_z, logvar_z, mu_x, logvar_x = mod(x = G_batch.pos, edge_index = G_batch.edge_index, batch = G_batch.batch)
        
        data = G_batch.pos.view(mu_z.shape[0],params['x_dim']*n_atoms)
        loss, kld_loss, mse_loss, logvar_loss = VAEloss(data, mu_x, logvar_x, mu_z, logvar_z, kl_weight = kl_weight)
        
        # backward pass
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        # accumulate losses
        ep_loss += loss.item()
        ep_kld_loss += kld_loss.item()
        ep_mse_loss += mse_loss.item()
        ep_logvar_loss += logvar_loss.item()
    
    # training diagnostics
    print('Epoch ' + str(ep+1) + ' Loss: ' + str(ep_loss))
    running_loss.append(ep_loss)
    running_kld_loss.append(ep_kld_loss)
    running_mse_loss.append(ep_mse_loss)
    running_logvar_loss.append(ep_logvar_loss)
    
    # dihedral plots
    
    if (ep + 1) % 100 == 0:
        T = 10000
        traj_ancestral = mod.ancestral_sample(T)
        traj_means = mod.sample_decoder_means(T)
    
        traj_ancestral = traj_ancestral.reshape((T,n_atoms, params['x_dim']))
        traj_means = traj_means.reshape((T,n_atoms, params['x_dim']))
    
        dangles_ancestral = md.compute_dihedrals(md.Trajectory(traj_ancestral,topology = traj.topology),[psi_inds,phi_inds])
        dangles_means = md.compute_dihedrals(md.Trajectory(traj_means, topology = traj.topology), [psi_inds,phi_inds])
    
        rng = [[-np.pi,np.pi],[-np.pi, np.pi]]
        bins = 40
        fig, ax = plt.subplots(1, 3, sharex = True, sharey = True)
    
        plt_name = 'dangles' + str(ep+1) + '.png'
        ax[0].hist2d(tr_dangles[:,1], tr_dangles[:,0], bins = bins, range = rng)
        ax[0].set_xlabel('$\phi$ / rad')
        ax[0].set_ylabel('$\psi$ / rad')
        ax[0].set_title('Training')
        ax[0].set_aspect('equal')
    
        ax[1].hist2d(dangles_means[:,1], dangles_means[:,0], bins = bins, range = rng)
        ax[1].set_title('MLE')
        ax[1].set_aspect('equal')
    
        ax[2].hist2d(dangles_ancestral[:,1], dangles_ancestral[:,0], bins = bins, range = rng)
        ax[2].set_title('Ancestral')
        ax[2].set_aspect('equal')
    
    
        plt.savefig(plt_name)
        plt.show()
        plt.close()
        
        # rog plot
        
        rog_T = subset_size
        traj_ancestral = mod.ancestral_sample(T)
        traj_means = mod.sample_decoder_means(T)
        
        traj_ancestral = traj_ancestral.reshape((T,n_atoms, params['x_dim']))
        traj_means = traj_means.reshape((T,n_atoms, params['x_dim']))
        
        rog_ancestral = [radius_of_gyration(xyz, topology = topology) for xyz in traj_means]
        rog_means = [radius_of_gyration(xyz, topology = topology) for xyz in traj_ancestral]
        
        plt.hist(tr_rogs, density = True, histtype = 'step', label = "Training")
        plt.hist(rog_ancestral, density = True, histtype = 'step', label = "Ancestral")
        plt.hist(rog_means, density = True, histtype = 'step', label = "MLE")
        plt.legend()
        
        plt_name = 'rogs' + str(ep+1) + '.png'
        plt.savefig(plt_name)
        plt.show()
        plt.close()

        


plt_name = 'loss.png'    
plt.plot(running_loss, label = 'Total Loss')
plt.plot(running_kld_loss, label = 'KLD Loss')
plt.plot(running_mse_loss, label = 'Recon Loss')
plt.plot(running_logvar_loss, label = 'Log Var Loss')
plt.legend()
plt.xlabel('Epoch')
plt.savefig(plt_name)
plt.show()
plt.close()

os.chdir(model_dir)
torch.save(mod.state_dict(), 'mod_64_32_silu_tanh.pth')


