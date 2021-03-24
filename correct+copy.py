import os
import shutil

for root, dirs, files in os.walk(r'E:\TableBank-Recognition\Recognition\annotations'):
    for file in files:
        name = file[:-4]
        shutil.copy(os.path.join(r'E:\TableBank-Recognition\Recognition\images', name + '.png'),
                    os.path.join(r'E:\TableBank-Recognition\Recognition\images_correct', name + '.png'))