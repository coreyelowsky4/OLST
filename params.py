# output directory
OUTPUT_DIRECTORY = './output/'

# input directories
RAW_VOLUME_PATH = '/mnt/brainstore8/anarasim/data/190327_Emxcre_Reconstruction/100pc/'
SEG_VOLUME_PATH = '/mnt/brainstore8/anarasim/data/190327_Emxcre_Reconstruction/100pc/'
FILE_EXTENSION = '.tif'

# resolution 
X_RES_OBLIQUE = .406
Y_RES_OBLIQUE = .406
Z_RES_OBLIQUE = 2.5

X_RES_CORONAL = .406
Y_RES_CORONAL = 2.5
Z_RES_CORONAL = .406

# clustering
CLUSTER_INTENSITY_THRESHOLD = 1000
CLUSTER_RADIUS_THRESHOLD = 30

# cropping
CROP_RADIUS = 25

# cnn classifier
CNN_MODEL_PATH = './cnn_model.json'
CNN_WEIGHTS_PATH = './cnn_weights.hdf5'
IMAGE_NORM_MAX_VALUE = 255
USE_CPU = True

# remove duplicate somas
STITCHING_PARAMETERS_PATH = './stitching_parameters.xml'
DISTANCE_THRESHOLD_DUPLICATES = 100

# remove close somas



