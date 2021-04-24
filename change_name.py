import os
from utils import get_file_name

for root, dirs, files in os.walk(r'E:\ICDAR19_TRACK_A\annotations'):
    idx = 0
    for file in files:
        filename = get_file_name(file)
        new_filename = str(idx)
        os.rename(os.path.join(root, filename + '.xml'), os.path.join(root, new_filename + '.xml'))
        os.rename(os.path.join(r'E:\ICDAR19_TRACK_A\images', filename + '.jpg'),
                  os.path.join(r'E:\ICDAR19_TRACK_A\images', new_filename + '.jpg'))
        idx += 1
