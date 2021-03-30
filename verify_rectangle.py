import cv2
import os

PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\Recognition\images'

with open('detect_v1/rec_deet.txt') as f:
    old_list = f.readlines()
    old_list = [line.strip() for line in old_list]

with open('detect_v1/wrong.txt') as f:
    wrong_list = f.readlines()
    wrong_list = [line.strip() for line in wrong_list]
    for line in wrong_list:
        old_list.remove(line)

counter = 0
new_list = []
with open('rec_deet.txt') as f:
    flist = f.readlines()
    flist = [line.strip() for line in flist]
    for file in flist:
        if file in old_list:
            counter += 1
            new_list.append(file)
            continue
        img = cv2.imread(os.path.join(PATH_TO_IMAGE_FOLDER, file))
        cv2.imshow('FFF', img)
        q = cv2.waitKey()
        if q != 32:
            print(file)
        else:
            new_list.append(file)
        cv2.destroyAllWindows()

print(counter)
with open('correct.txt', 'w') as f:
    for item in new_list:
        f.write(item + '\n')
