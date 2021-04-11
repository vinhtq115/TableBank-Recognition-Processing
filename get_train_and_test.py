import os
from shutil import copyfile
from random import sample

PATH_TO_ANNOTATION_FOLDER = r'E:\TableBank-Recognition\final_recognition_data\annotations'
PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\final_recognition_data\images'
PATH_TO_ANNOTATION_FOLDER_2 = r'E:\TableBank-Recognition\recognition_sampled\annotations'
PATH_TO_IMAGE_FOLDER_2 = r'E:\TableBank-Recognition\recognition_sampled\images'

with open(r'E:\TableBank-Recognition\final_recognition_data\correct_files.txt') as f:
    correct_files = f.readlines()
    correct_files = [i.strip() for i in correct_files]
    correct_files_sample = sample(correct_files, 4500)
    for file in correct_files_sample:
        copyfile(os.path.join(PATH_TO_IMAGE_FOLDER, file + '.png'), os.path.join(PATH_TO_IMAGE_FOLDER_2, file + '.png'))
        copyfile(os.path.join(PATH_TO_ANNOTATION_FOLDER, file + '.xml'), os.path.join(PATH_TO_ANNOTATION_FOLDER_2, file + '.xml'))

with open(r'E:\TableBank-Recognition\final_recognition_data\wrong_files.txt') as f:
    wrong_files = f.readlines()
    wrong_files = [i.strip() for i in wrong_files]
    wrong_files_sample = sample(wrong_files, 500)
    for file in wrong_files_sample:
        copyfile(os.path.join(PATH_TO_IMAGE_FOLDER, file + '.png'), os.path.join(PATH_TO_IMAGE_FOLDER_2, file + '.png'))
        copyfile(os.path.join(PATH_TO_ANNOTATION_FOLDER, file + '.xml'), os.path.join(PATH_TO_ANNOTATION_FOLDER_2, file + '.xml'))
