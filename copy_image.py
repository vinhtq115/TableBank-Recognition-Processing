import os
from shutil import copyfile
from utils import get_file_name

IMAGE_DIR = r'E:\TableBank-Recognition\final_recognition_data\images'
ORIGINAL_IMAGE_DIR = r'E:\TableBank-Recognition\Recognition\images'
ANNOTATION_DIR = r'E:\TableBank-Recognition\final_recognition_data\annotations'


if __name__ == '__main__':
    for _, _, files in os.walk(ANNOTATION_DIR):
        for file in files:
            filename = get_file_name(file)
            copyfile(os.path.join(ORIGINAL_IMAGE_DIR, filename + '.png'), os.path.join(IMAGE_DIR, filename + '.png'))
