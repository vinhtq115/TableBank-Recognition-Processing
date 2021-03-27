import os
import shutil
from utils import get_file_name


for root_ann, _, files_ann in os.walk(r'E:\TableBank-Recognition\Recognition\final\annotations'):
    finised_files_ann = [get_file_name(x) for x in files_ann]
    for file in finised_files_ann:
        shutil.copy(os.path.join(r'E:\TableBank-Recognition\Recognition\images', file + '.png'),
                    os.path.join(r'E:\TableBank-Recognition\Recognition\final\images', file + '.png'))
