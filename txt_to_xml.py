"""
Read original TableBank Recognition annotation files and save them to XML format individually.
"""

import os
import glob
from bs4 import BeautifulSoup

# Path to original annotation files (*.txt)
ORIGINAL_ANNOTATION_PATH = r'E:\TableBank-Recognition\Recognition\annotations_original'
# Path to save processed XML annotation files
PROCESSED_ANNOTATION_PATH = r'E:\TableBank-Recognition\Recognition\annotations'

for name_list in glob.glob(os.path.join(ORIGINAL_ANNOTATION_PATH, 'all*.txt')):
    which = name_list.split('\\')[-1].split('.')[0]  # Check if reading all_test, all_train or all_val

    # Read file name list in all_*.txt
    f_name_list = open(name_list).readlines()
    # Read annotations in tgt-all_*.txt
    f_annotation_list = open(os.path.join(ORIGINAL_ANNOTATION_PATH, 'tgt-' + which + '.txt')).readlines()
    assert f_name_list.__len__() == f_annotation_list.__len__()  # Make sure length of 2 lists are same

    # Save annotations as XML
    for file_name, file_annotation in zip(f_name_list, f_annotation_list):
        with open(os.path.join(PROCESSED_ANNOTATION_PATH, file_name.rstrip() + '.txt'), 'w') as f:
            soup = BeautifulSoup(file_annotation.replace('tdy', 'tdy/').replace('tdn', 'tdn/'), 'xml')
            f.write(soup.prettify())
