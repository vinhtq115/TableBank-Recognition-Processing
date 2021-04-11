import os
from utils import get_file_name

for root, dirs, files in os.walk(r'E:\TableBank-Recognition\recognition_sampled\annotations'):
    idx = 0
    for file in files:
        filename = get_file_name(file)
        new_filename = str(idx)
        os.rename(os.path.join(root, filename + '.xml'), os.path.join(root, new_filename + '.xml'))
        os.rename(os.path.join(r'E:\TableBank-Recognition\recognition_sampled\images', filename + '.png'),
                  os.path.join(r'E:\TableBank-Recognition\recognition_sampled\images', new_filename + '.png'))
        idx += 1