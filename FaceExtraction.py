#%% Import packages
import os
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray


#%% Define some params
SOURCE_BASE_PATH = "Face_Dataset"
TARGET_BASE_PATH = "Extracted_Faces"


#%% function: extract_face()
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
	# load image from file
	image = Image.open(filename)

  # convert to RGB, if needed
	image = image.convert('RGB')

  # convert to array
	pixels = asarray(image)

  # create the detector, using default weights
	detector = MTCNN()

  # detect faces in the image
	results = detector.detect_faces(pixels)

  # extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']

  # bug fix
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height

  # extract the face
	face = pixels[y1:y2, x1:x2]

  # resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)

	return face_array

#%% Main
idx = 0
for cur_path in os.listdir(SOURCE_BASE_PATH):
    src_sub_path = os.path.join(SOURCE_BASE_PATH, cur_path)
    tgt_sub_path = os.path.join(TARGET_BASE_PATH, cur_path)

    print(src_sub_path)
    print(tgt_sub_path)

    if not os.path.isdir(tgt_sub_path):
        os.mkdir(tgt_sub_path)

    jdx = 0
    for filename in os.listdir(src_sub_path):
        print("Progress - idx: %s / %03d" % (cur_path, jdx))
        src_file_path = os.path.join(src_sub_path, filename)
        pixels = extract_face(src_file_path)
        img = Image.fromarray(pixels, mode='RGB')
        img.save(os.path.join(tgt_sub_path, filename))

        jdx += 1

    idx += 1
