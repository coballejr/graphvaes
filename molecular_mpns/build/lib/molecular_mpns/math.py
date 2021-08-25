import numpy as np

def radius_of_gyration(xyz, topology):
    masses = np.array([atom.element.mass for atom in topology.atoms])
    total_mass = masses.sum()
    com = np.array([m*x for m,x in zip(masses,xyz)]).sum(axis = 0) / total_mass
    
    r_sq = ((xyz - com)**2).sum(axis = 1) 
    rog_sq = (masses*r_sq).sum() / total_mass
    
    return np.sqrt(rog_sq)
