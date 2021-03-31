import os

with open(r'E:\TableBank-Recognition\Recognition\flows\List of files with missing font (rectangles).txt') as f:
    wrong_list = f.readlines()
    wrong_list = [line.strip().replace('.png', '.xml') for line in wrong_list]
    wrong_set = set(wrong_list)
for root, _, files in os.walk(r'E:\TableBank-Recognition\final_recognition_data\annotations'):
    to_be_deleted = [file for file in files if file in wrong_set]
    for file in to_be_deleted:
        os.remove(os.path.join(root, file))
