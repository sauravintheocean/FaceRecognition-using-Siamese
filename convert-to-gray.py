#%% Import libraries
import os
import cv2


#%% Main
# load the photo and extract the face
#src_base_path = "./Extracted_Faces"
#tgt_base_path = "./Gray_Faces_Updated"
src_base_path = "Extracted_Faces"
tgt_base_path = "Gray_Faces"


idx = 0
for cur_path in os.listdir(src_base_path):
    src_sub_path = os.path.join(src_base_path, cur_path)
    tgt_sub_path = os.path.join(tgt_base_path, cur_path)

    if not os.path.isdir(tgt_sub_path):
        os.mkdir(tgt_sub_path)

    jdx = 0
    for filename in os.listdir(src_sub_path):
        filepath = os.path.join(src_sub_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        target_path = os.path.join(tgt_sub_path, "%03d.jpg" % (jdx))
        print(target_path)
        cv2.imwrite(target_path, img)
        jdx += 1
