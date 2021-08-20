from torch_geometric.data import Data
from torch import tensor, cdist, triu_indices
from itertools import product


def fully_connected_adjacency_matrix(N):
    inds_i, inds_j = [], []
    for i, j in product(range(N), range(N)):
        inds_i.append(i)
        inds_j.append(j)
    edge_index = tensor([inds_i,inds_j]).long()
    
    return edge_index

def bond_adjacency_matrix(topology):
    edges_to = [[b[0].index, b[1].index] for b in topology.bonds]
    edges_from = [[b[0].index, b[1].index] for b in topology.bonds]
    
    edges = edges_to + edges_from
    edge_index = tensor(edges).long()
    
    return edge_index.t()
    
    

class AlanineDipeptideGraph(Data):
    
    def __init__(self, z, pos, edge_index = None, dists = None):
        '''
        
        Parameters
        ----------
        z : atomic charges, long tensor, shape = (n_atoms,).
        pos : atomic positions, tensor, shape = (n_atoms,3)
        Returns
        -------
        None.
        '''
        super(AlanineDipeptideGraph,self).__init__()
        self.z = z
        self.pos = pos
        
        if edge_index is not None:
            self.edge_index = edge_index
            
        
    def compute_dists(self):
        
        self.dists = cdist(self.pos, self.pos)
        
        n_atoms = self.dists.shape[0]
        triu_inds = triu_indices(n_atoms, n_atoms, 1)
        self.triu_dists = self.dists[triu_inds[0], triu_inds[1]]
      
            
        return self
        
            
            
