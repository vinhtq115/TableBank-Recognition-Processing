"""
Sample X amount of images, as well as copy their annotation file.
"""

import os
import glob
import random
from shutil import copyfile

ORIGINAL_IMAGE_PATH = r'E:\TableBank-Recognition\Recognition\images'
SAMPLE_IMAGE_PATH = r'E:\TableBank-Recognition\sampled\images'
ORIGINAL_ANNOTATION_PATH = r'E:\TableBank-Recognition\Recognition\annotations'
SAMPLE_ANNOTATION_PATH = r'E:\TableBank-Recognition\sampled\annotations'
NUMBER_OF_SAMPLES = 1000

a = glob.glob(os.path.join(ORIGINAL_IMAGE_PATH, '*.png'))
sampled = random.sample(a, NUMBER_OF_SAMPLES)

for file in sampled:
    file_name = file.split('\\')[-1]
    file_name = file_name[:file_name.rfind('.')]
    copyfile(file, os.path.join(SAMPLE_IMAGE_PATH, file.split('\\')[-1]))
    copyfile(os.path.join(ORIGINAL_ANNOTATION_PATH, file_name + '.xml'),
             os.path.join(SAMPLE_ANNOTATION_PATH, file_name + '.xml'))
