import os
from utils import get_file_name

with open(r'E:\TableBank-Recognition\Recognition\flow1.txt') as f1:
    finised_files1 = f1.readlines()
    finised_files1 = [x.strip() for x in finised_files1]
    count = 0
    for root, dirs, files in os.walk(r'E:\TableBank-Recognition\Recognition\annotations'):
        for file in files:
            if get_file_name(file) not in finised_files1:
                count += 1
                os.remove(os.path.join(root, file))
    print(count)
