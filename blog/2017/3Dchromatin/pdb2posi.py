from utils import *
import numpy as np

def pdb2posi(filename='chr1.pdb'):
	# extract coordinates from pdb file
    position = []
    for line in open(filename):
        parse = line.split()
        if parse[0] == 'ATOM' and parse[2] == 'CA':
            position.append(parse[5:8])

    num_bead = len(position)
    location = np.empty((num_bead, 3), dtype=float)
    for i in range(num_bead):
        for j in range(3):
            location[i][j] = float(position[i][j])
    return location

location = pdb2posi()
dist = pairwise_dist(location)