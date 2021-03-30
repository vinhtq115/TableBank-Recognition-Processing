import os

flow1 = r'E:\TableBank-Recognition\Recognition\flows\flow1'
flow2 = r'E:\TableBank-Recognition\Recognition\flows\flow2'
flow3 = r'E:\TableBank-Recognition\Recognition\flows\flow3'

files = []
for root1, _, files1 in os.walk(flow1):
    for file in files1:
        files.append(file)
for root2, _, files2 in os.walk(flow2):
    for file in files2:
        if file not in files:
            files.append(file)
for root3, _, files3 in os.walk(flow3):
    for file in files3:
        if file not in files:
            files.append(file)
print(len(files))
