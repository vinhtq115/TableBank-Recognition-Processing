import os
from utils import get_file_name

PATH_TO_DESTINATION_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\annotations'
PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\Recognition\images'

TXT_FLOW1 = r'E:\TableBank-Recognition\Recognition\flow1.txt'
TXT_FLOW2 = r'E:\TableBank-Recognition\Recognition\flow2_ocr.txt'
FLOW2_INCORRECT = r'E:\TableBank-Recognition\Recognition\flow2_incorrect.txt'
FLOW2_UNKNOWN = r'E:\TableBank-Recognition\Recognition\flow2_unknown.txt'

count = 0
with open(TXT_FLOW2) as f2:
    finised_files2 = f2.readlines()
    finised_files2 = [x.strip() for x in finised_files2]
    with open(FLOW2_INCORRECT) as f2_incorrect:
        f2_incorrect_files = f2_incorrect.readlines()
        f2_incorrect_files = [x.strip() for x in f2_incorrect_files]
        with open(FLOW2_UNKNOWN, 'w') as f2_unknown:
            for root, dirs, files in os.walk(PATH_TO_IMAGE_FOLDER):
                for file in files:
                    file_name = get_file_name(file)
                    if file_name not in finised_files2 and file_name not in f2_incorrect_files:
                        count += 1
                        f2_unknown.write(file_name + '\n')
                print(count)
