import xml.etree.ElementTree as ET
import numpy as np


# given two volume ids return whether the volumes are adjacent
def is_adjacent(volume_a, volume_b):
	
	z_a = int(volume_a[1:3])
	z_b = int(volume_b[1:3])

	y_a = int(volume_a[5:])
	y_b = int(volume_b[5:])
	
	if abs(z_a-z_b) <= 1 and abs(y_a-y_b) <= 1:

		return True
	else:
		return False

def extract_stitching_parameters(xml_file):

	# read in xml
	tree = ET.parse(xml_file)
	root = tree.getroot() 

	# output lists
	files = []
	registrations = []
	stitchings = []

	# find nodes for files, registrations, stitchings
	for child in root.iter():
		if child.tag == 'files':
			files_node = child
		elif child.tag == 'ViewRegistrations':
			registrations_node = child
		elif child.tag == 'StitchingResults':
			stitchings_node = child

	for child in files_node:
		setup_number = child.attrib['view_setup']
		file_name = child[0].text
		dict_data = {"setup number":setup_number,"file name":file_name}
		files.append(dict_data)
	
	for child in registrations_node:
		setup_number = child.attrib['setup']
		stitching_transform = np.fromstring(child[0][1].text, sep=' ')
		translation_regular_grid = np.fromstring(child[1][1].text, sep=' ')
		calibration = np.fromstring(child[2][1].text, sep=' ')
		dict_data = {"setup number":setup_number,"stitching transform":stitching_transform,"translation to regular grid":translation_regular_grid,"calibration":calibration}
		registrations.append(dict_data)

	for child in stitchings_node:
		setup_a_number = child.attrib['view_setup_a']
		setup_b_number = child.attrib['view_setup_b']	
		shift = np.fromstring(child[0].text, sep=' ')
		bounding_box = np.fromstring(child[3].text, sep=' ')
		dict_data = {"setup number a":setup_a_number,"setup number b":setup_b_number,"shift":shift,"bounding box":bounding_box}
		stitchings.append(dict_data)

	return files, registrations, stitchings

def get_stitching_matrices(files,registrations,volume):

	# Get associated setup id
	setup_id = -1
	for f in files:
		if volume in f['file name']:
			setup_id = f['setup number']
			break

	# Make sure volume was found in XML
	if setup_id == -1:
		print("Error: Volume name not found in XML")
		return -1

	# Get associated stitching parameters
	stitch_params = -1
	for r in registrations:
		if r['setup number'] == setup_id:
			stitch_params = r
			break

	# Make sure volume was found in XML
	if stitch_params == -1:
		print("Error: Volume stitching parameters name not found in XML")
		return -1

	# Extract affine transformation matrices and add fourth row (matrices are 3x4)
	translation_to_grid_matrix = np.vstack([np.reshape(stitch_params['translation to regular grid'], (3,4)),np.array([0,0,0,1])])
	stitching_matrix = np.vstack([np.reshape(stitch_params['stitching transform'], (3,4)),np.array([0,0,0,1])])
	calibration_matrix = np.vstack([np.reshape(stitch_params['calibration'], (3,4)),np.array([0,0,0,1])])

	return translation_to_grid_matrix, stitching_matrix, calibration_matrix
