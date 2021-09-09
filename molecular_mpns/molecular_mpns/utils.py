import mdtraj

""" This function will create a topology object for visualization:"""
def create_top(N):
    # Initialize topology object:
    top = mdtraj.Topology()
    carbon = mdtraj.element.carbon
    ch1 = top.add_chain()
    # Add a residue:
    res1 = top.add_residue("RES1", ch1)
    # Add all atoms:
    atoms = []
    for ii in range(N):
        at_ii = top.add_atom(name="C%d"%ii, element=carbon, residue=res1)
        atoms.append(at_ii)
    # Add bonds:
    for ii in range(N-1):
        top.add_bond(atoms[ii], atoms[ii+1])
    return top