import numpy as np
import os
from sklearn.metrics import pairwise_distances
import itertools
import params
import utils
import copy

OBLIQUE_RES = np.array([params.X_RES_OBLIQUE,params.Y_RES_OBLIQUE,params.Z_RES_OBLIQUE])

# load somas
somas = np.load(os.path.join(params.OUTPUT_DIRECTORY,'cnn_predicted_somas.npy'),allow_pickle=True)

# convert volume coords to stitching coords
stitching_somas = np.empty_like(somas)
stitching_somas[:,0] = somas[:,0]

files, registrations, stitchings = utils.extract_stitching_parameters(params.STITCHING_PARAMETERS_PATH)

for i,soma in enumerate(somas):
	
	volume = soma[0]
	soma_volume_coords = soma[1:]

	# get tranformation matrices
	translation_to_grid_matrix, stitching_matrix, calibration_matrix = utils.get_stitching_matrices(files,registrations,volume)

	# apply transformtaion
	soma_stitching_coords = translation_to_grid_matrix @ stitching_matrix @ calibration_matrix @ np.append(soma_volume_coords,1).T
	
	stitching_somas[i,1:] = soma_stitching_coords[:-1]

# calculate pairwise distances
distances_matrix = pairwise_distances(stitching_somas[:,1:]) * OBLIQUE_RES[0]

# set all somas over threshold to Inf
distances_matrix[distances_matrix > params.DISTANCE_THRESHOLD_DUPLICATES] = np.Inf

# set all same volume somas to Inf
for unique_vol in np.unique(somas[:,0]):
	idxs = np.where(somas[:,0] == unique_vol)[0]
	for idx_a in idxs:
		for idx_b in idxs:
			distances_matrix[idx_a,idx_b] = np.Inf

# set all non adjacent somas to Inf
for idx_a in range(len(somas)):
	for idx_b in range(idx_a+1,len(somas)):
		if not utils.is_adjacent(somas[idx_a,0],somas[idx_b,0]):
			distances_matrix[idx_a,idx_b] = np.Inf
			distances_matrix[idx_b,idx_a] = np.Inf
		
# iterate until all possible duplicates are exhausted 

duplicates = []
num_inf = np.sum(distances_matrix != np.Inf)

while num_inf > 0:
	
	# temporary set to keep track of connections to remove
	temp_set = set()

	# get minimum
	idx_a,idx_b = np.unravel_index(distances_matrix.argmin(), distances_matrix.shape)

	# check if one of idxs already in a duplicates set
	# then add other soma to set
	new_set = True
	for i,dup_set in enumerate(duplicates):
		if idx_a in dup_set:
			dup_set.add(idx_b)
			temp_set = dup_set
			new_set = False
			break
		if idx_b in dup_set:
			dup_set.add(idx_a)
			temp_set = dup_set
			new_set = False
			break

	# add both indicies to temp set
	temp_set.add(idx_a)
	temp_set.add(idx_b)

	# remove possible futures connections between somas in volumes
	for pair in list(itertools.combinations(temp_set,2)):
		a,b = pair[0],pair[1]	
		vol_a, vol_b = somas[a,0], somas[b,0]
		vol_a_idxs, vol_b_idxs = np.where(somas[:,0] == vol_a)[0], np.where(somas[:,0] == vol_b)[0]
		distances_matrix[a,vol_b_idxs] = np.Inf
		distances_matrix[vol_b_idxs,a] = np.Inf
		distances_matrix[b,vol_a_idxs] = np.Inf
		distances_matrix[vol_a_idxs,b] = np.Inf

	# add to duplicates list
	if new_set:
		duplicates.append({idx_a,idx_b})

	num_inf = np.sum(distances_matrix != np.Inf)

# create output
distances_matrix_duplicates = np.zeros(shape=(0,16),dtype=object)
already_added = set()

for i,soma in enumerate(somas):

	# check if somas is in duplicates
	in_duplicates = [d for d in duplicates if i in d]

	if len(in_duplicates) == 0 and i not in already_added:
		soma_to_add = np.concatenate((soma,np.full(12,None)))
		distances_matrix_duplicates = np.vstack((distances_matrix_duplicates,soma_to_add))
	elif len(in_duplicates) > 0 :
		d = in_duplicates[0]
		duplicates.remove(d)
		for x in d:
			already_added.add(x)
		somas_to_add = np.concatenate((np.concatenate([somas[x] for x in d]),np.full(16-4*len(d),None)))
		distances_matrix_duplicates = np.vstack((distances_matrix_duplicates,somas_to_add))

# save soma
np.save(os.path.join(params.OUTPUT_DIRECTORY,'removed_duplicate_somas'),distances_matrix_duplicates)
np.savetxt(os.path.join(params.OUTPUT_DIRECTORY,'removed_duplicate_somas.txt'),distances_matrix_duplicates,delimiter=' ',fmt="%s")











