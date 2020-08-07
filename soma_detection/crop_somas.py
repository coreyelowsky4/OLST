import os
import numpy as np
from skimage.io import imread
import params

print()
print('###########')
print('Crop Somas')
print('###########')
print()

print('# Parameters #')
print('Raw Volume Path:',params.RAW_VOLUME_PATH)
print('Segmented Volume Path:',params.SEG_VOLUME_PATH)
print('Crop Radius:',params.CROP_RADIUS,'(um)')
print()

# define resolution array
OBLIQUE_RES = np.array([params.X_RES_OBLIQUE,params.Y_RES_OBLIQUE,params.Z_RES_OBLIQUE])

# load somas
soma_path = os.path.join(params.OUTPUT_DIRECTORY,'clustered_somas.npy')
somas = np.load(soma_path, allow_pickle=True)

print('# Somas:',len(somas))
print()
 
# get cropping radii
crop_radii = (params.CROP_RADIUS/OBLIQUE_RES).astype(np.uint16)
cropped_shape = (2*crop_radii[2],2*crop_radii[1],2*crop_radii[0])

# array to store cropped images
cropped_raw = np.empty(shape=(len(somas),cropped_shape[0],cropped_shape[1],cropped_shape[2]),dtype=np.uint16)
cropped_seg = np.empty(shape=(len(somas),cropped_shape[0],cropped_shape[1],cropped_shape[2]),dtype=np.uint16)

# get unique volumes
unique_volumes = np.unique(somas[:,0])

# to index into cropped arrays
idx = 0

# iterate through unique volumes
for volume in unique_volumes:

	print('Volume:', volume)

	# get somas in volume
	volume_somas = somas[somas[:,0] == volume][:,1:]

	# load raw and segmented volume
	print('Load Raw Volume...')
	raw_volume = imread(os.path.join(params.RAW_VOLUME_PATH,volume+params.FILE_EXTENSION))
	print('Load Segmented Volume...')
	seg_volume = imread(os.path.join(params.SEG_VOLUME_PATH,volume+params.FILE_EXTENSION))
	
	# iterate through somas
	for soma in volume_somas:

		# get bounds to crop
		min_coords = soma - crop_radii
		max_coords = soma + crop_radii

		# fix boundary issues
		min_coords[min_coords < 0] = 0

		# crop raw and segmented soma
		cropped_raw_soma = raw_volume[min_coords[2]:max_coords[2],min_coords[1]:max_coords[1],min_coords[0]:max_coords[0]]
		cropped_seg_soma = seg_volume[min_coords[2]:max_coords[2],min_coords[1]:max_coords[1],min_coords[0]:max_coords[0]]

		# pad if needed
		if cropped_raw_soma.shape != cropped_shape:
			x,y,z = cropped_shape[2],cropped_shape[1],cropped_shape[0]
			cropped_raw_soma = np.pad(cropped_raw_soma,((0,z-cropped_raw_soma.shape[0]),(0,y-cropped_raw_soma.shape[1]),(0,x-cropped_raw_soma.shape[2])),'constant',constant_values=np.amin(cropped_raw_soma))
			cropped_seg_soma = np.pad(cropped_seg_soma,((0,z-cropped_seg_soma.shape[0]),(0,y-cropped_seg_soma.shape[1]),(0,x-cropped_seg_soma.shape[2])),'constant',constant_values=np.amin(cropped_seg_soma))

		# store cropped volumes
		cropped_raw[idx,:,:,:] = cropped_raw_soma
		cropped_seg[idx,:,:,:] = cropped_seg_soma
	
		idx += 1

	# save cropped somas
	np.save(os.path.join(params.OUTPUT_DIRECTORY,'cropped_raw_somas'),cropped_raw)
	np.save(os.path.join(params.OUTPUT_DIRECTORY,'cropped_seg_somas'),cropped_seg)





		






	
