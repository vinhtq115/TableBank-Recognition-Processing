import os
from shutil import copyfile

IMAGE_DIR = r'E:\TableBank-Recognition\final_recognition_data\images'
ORIGINAL_IMAGE_DIR = r'E:\TableBank-Recognition\Recognition\images'
ANNOTATION_DIR = r'E:\TableBank-Recognition\final_recognition_data\annotations'
FLOW1_ANN = r'E:\TableBank-Recognition\Recognition\flows\flow1'
FLOW3_ANN = r'E:\TableBank-Recognition\Recognition\flows\flow3'
FLOW4_ANN = r'E:\TableBank-Recognition\Recognition\flows\flow4'


if __name__ == '__main__':
    for _, _, flow1_ann in os.walk(FLOW1_ANN):
        for file in flow1_ann:
            copyfile(os.path.join(FLOW1_ANN, file), os.path.join(ANNOTATION_DIR, file))
    for _, _, flow3_ann in os.walk(FLOW3_ANN):
        for file in flow3_ann:
            copyfile(os.path.join(FLOW3_ANN, file), os.path.join(ANNOTATION_DIR, file))
    for _, _, flow4_ann in os.walk(FLOW4_ANN):
        for file in flow4_ann:
            copyfile(os.path.join(FLOW4_ANN, file), os.path.join(ANNOTATION_DIR, file))
