import os
import json
from utils import get_file_name

ROOT_FOLDER = r'E:\ICDAR2019_test'  # Root folder
PATH_TO_IMAGES = os.path.join(ROOT_FOLDER, 'images')  # Images folder
PATH_TO_PREDICT = os.path.join(ROOT_FOLDER, 'final_result_0.7_nonoverlap.json')  # Prediction result
PATH_TO_RESULT = os.path.join(ROOT_FOLDER, 'result_table_0.7_nonoverlap')

# Load prediction
with open(PATH_TO_PREDICT) as f:
    prediction = json.load(f)

for file in prediction.keys():
    filename = get_file_name(file)
    result = ''
    destination_file = os.path.join(PATH_TO_RESULT, filename + '.txt')

    for cell in prediction[file]:
        result += 'table_cell ' + cell['confidence'] + ' ' + str(cell['min_x']) + ' ' + str(cell['min_y']) + ' ' + \
                  str(cell['max_x']) + ' ' + str(cell['max_y']) + '\n'

    with open(destination_file, 'w') as f:
        f.write(result)
