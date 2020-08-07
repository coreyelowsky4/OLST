import numpy as np
import params
import os
from keras.models import model_from_json
import tensorflow as tf

print()
print('# Parameters #')
print('CNN MODEL:',params.CNN_MODEL_PATH)
print('CNN WEIGHTS:',params.CNN_WEIGHTS_PATH)
print('NORMALIZATION MAX VALUE:',params.IMAGE_NORM_MAX_VALUE)
print()

### Fix Errors and Warnings ####
if params.USE_CPU:
	os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
################################


print('Load Data...')
cropped_raw = np.load(os.path.join(params.OUTPUT_DIRECTORY,'cropped_raw_somas.npy')).astype(float)
cropped_seg = np.load(os.path.join(params.OUTPUT_DIRECTORY,'cropped_seg_somas.npy')).astype(float)


print('Normalize Images...')
for i,image in enumerate(cropped_raw):
	cropped_raw[i] = params.IMAGE_NORM_MAX_VALUE*(image - np.amin(image)) / (np.amax(image) - np.amin(image))

for i,image in enumerate(cropped_seg):
	cropped_seg[i] = params.IMAGE_NORM_MAX_VALUE*(image - np.amin(image)) / (np.amax(image) - np.amin(image))

# cast back to int and place into array for input into CNN
cropped_raw = cropped_raw.astype(np.uint8)
cropped_seg = cropped_seg.astype(np.uint8)
X = np.zeros(shape=(len(cropped_raw),cropped_raw.shape[1],cropped_raw.shape[2],cropped_raw.shape[3],2),dtype=np.uint8)
X[:,:,:,:,0] = cropped_raw
X[:,:,:,:,1] = cropped_seg

print('Load Model...')	
with open(params.CNN_MODEL_PATH, 'r') as fp:
	model_json = fp.read()

model = model_from_json(model_json)
print(model.summary())
model.load_weights(params.CNN_WEIGHTS_PATH)

print('Predict...')
y_pred = np.round(model.predict(X).flatten()).astype(np.uint8)

# load soma list
somas = np.load(os.path.join(params.OUTPUT_DIRECTORY,'clustered_somas.npy'),allow_pickle=True)
somas = somas[y_pred.astype(bool)]

# save somas
np.save(os.path.join(params.OUTPUT_DIRECTORY,'cnn_predicted_somas.npy'),somas)
np.savetxt(os.path.join(params.OUTPUT_DIRECTORY,'cnn_predicted_somas.csv'),somas,delimiter=' ',fmt='%s')

