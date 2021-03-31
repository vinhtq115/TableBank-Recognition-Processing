import os

detected_files = set()
for root, dirs, files in os.walk(r'E:\TableBank-Recognition\Recognition\flows\flow1'):
    for file in files:
        detected_files.add(file)

for root, dirs, files in os.walk(r'E:\TableBank-Recognition\Recognition\flows\flow2_1'):
    for file in files:
        detected_files.add(file)

for root, dirs, files in os.walk(r'E:\TableBank-Recognition\Recognition\flows\flow2_2'):
    for file in files:
        detected_files.add(file)

for root, dirs, files in os.walk(r'E:\TableBank-Recognition\Recognition\flows\flow3'):
    for file in files:
        detected_files.add(file)

for root, dirs, files in os.walk(r'E:\TableBank-Recognition\Recognition\flows\flow4'):
    for file in files:
        detected_files.add(file)

with open(r'E:\TableBank-Recognition\Recognition\flows\List of files with missing font (rectangles).txt') as f:
    wrong_list = f.readlines()
    wrong_list = [line.strip().replace('.png', '.xml') for line in wrong_list]
    wrong_set = set(wrong_list)
    print(len(detected_files - wrong_set))
