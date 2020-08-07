import os
import numpy as np
from skimage.io import imread
import params

# Clustering Algorithm to Detect Somas in Image Stack

print()
print('##########################')
print('DETECT SOMAS (CLUSTERING)')
print('##########################')
print()

print('# Parameters #')
print('Input Directory:',params.RAW_VOLUME_PATH)
print('Output Directory:',params.OUTPUT_DIRECTORY)
print('Input Volume File Extension:',params.FILE_EXTENSION)
print('Intensity Threshold:',params.CLUSTER_INTENSITY_THRESHOLD)
print('Radius Threshold:',params.CLUSTER_RADIUS_THRESHOLD,'(um)')
print()

# define resolution array
OBLIQUE_RES = np.array([params.X_RES_OBLIQUE,params.Y_RES_OBLIQUE,params.Z_RES_OBLIQUE])

# get all files in directory with appropriate extension
volumes = [f for f in os.listdir(params.RAW_VOLUME_PATH) if f.endswith(params.FILE_EXTENSION)]
volumes.sort()
num_volumes = len(volumes)
print('# Volumes:',num_volumes)
print()

# array to store soma centroids for all volumes
global_soma_centroids = np.empty(shape=(0,4),dtype=object)

# iterate through all volumes
for volume in volumes:

	volume_id = volume[:-4]

	print('Volume ID:',volume_id)
	
	print('Loading Volume...')
	volume_full_path = os.path.join(params.RAW_VOLUME_PATH,volume)
	raw_volume = imread(volume_full_path)

	print('Threshold by Intensity...')
	soma_voxels = np.argwhere(raw_volume > params.CLUSTER_INTENSITY_THRESHOLD)

	# concat column of intensities
	intensities = raw_volume[soma_voxels[:,0],soma_voxels[:,1],soma_voxels[:,2]]
	soma_voxels = np.hstack((soma_voxels,intensities.reshape(len(intensities),1)))

	# sort voxels in descending order of intensity
	print('Sort Voxels by Intensity...')
	soma_voxels = soma_voxels[soma_voxels[:,-1].argsort()[::-1]][:,:-1]

	# fix coordinate order
	soma_voxels = soma_voxels[:,[2,1,0]]

	# initialize centroids array
	# fourth column to hold count of voxels for centroid updates
	soma_centroids = np.empty(shape=(0,4))
	
	# iterate through all soma voxels in volume
	print('Cluster Somas...')
	for soma_voxel in soma_voxels:

		if len(soma_centroids) == 0:
			soma_centroids = np.vstack((soma_centroids,np.concatenate((soma_voxel,[1]))))
		else:
			
			# find closest centroid 
			soma_centroid_distances = np.sqrt(np.sum(((soma_centroids[:,:-1] - soma_voxel)*OBLIQUE_RES)**2,axis=1))
			min_index = np.argmin(soma_centroid_distances)
			min_distance = soma_centroid_distances[min_index]

			# comapre disance to radius threshold
			if min_distance > params.CLUSTER_RADIUS_THRESHOLD:
				# create new centroid
				soma_centroids = np.vstack((soma_centroids,np.concatenate((soma_voxel,[1]))))
			else:
				# update centroid
				soma_centroids[min_index,:-1] = (soma_centroids[min_index,:-1]*soma_centroids[min_index,-1] + soma_voxel) / (soma_centroids[min_index,-1] + 1)
				soma_centroids[min_index,-1] = soma_centroids[min_index,-1] + 1
				
	soma_centroids = np.round(soma_centroids[:,:-1]).astype(np.uint16)
	print('# Somas Detected in Volume:',len(soma_centroids))
	print()

	soma_centroids = np.hstack((np.array([volume_id]*len(soma_centroids),dtype=object).reshape(len(soma_centroids),1),soma_centroids))
	global_soma_centroids = np.vstack((global_soma_centroids,soma_centroids))

	# save soma
	np.save(os.path.join(params.OUTPUT_DIRECTORY,'clustered_somas'),global_soma_centroids)
	np.savetxt(os.path.join(params.OUTPUT_DIRECTORY,'clustered_somas.txt'),global_soma_centroids,delimiter=' ',fmt="%s")





