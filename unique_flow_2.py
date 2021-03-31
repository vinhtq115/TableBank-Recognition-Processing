flow2_1 = open(r'E:\TableBank-Recognition\Recognition\flow2_1.txt').readlines()
flow2_1 = [i.strip() for i in flow2_1]
f2_1 = set(flow2_1)
flow2_2 = open(r'E:\TableBank-Recognition\Recognition\flow2_2.txt').readlines()
flow2_2 = [i.strip() for i in flow2_2]
f2_2 = set(flow2_2)

print('Total files:' + str(len(f2_1 | f2_2)))
print('Same files:' + str(len(f2_1 & f2_2)))
print('Unique files of flow 2_1: ' + str(len(f2_1 - f2_2)))
print('Unique files of flow 2_2: ' + str(len(f2_2 - f2_1)))
