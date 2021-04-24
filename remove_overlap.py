import os
import json
from utils import get_file_name


def calculate_iou(bb1, bb2):
    assert bb1['min_x'] <= bb1['max_x']
    assert bb1['min_y'] <= bb1['max_y']
    assert bb2['min_x'] <= bb2['max_x']
    assert bb2['min_y'] <= bb2['max_y']

    # Coordinates of the intersection area
    x_left = max(bb1['min_x'], bb2['min_x'])
    y_top = max(bb1['min_y'], bb2['min_y'])
    x_right = min(bb1['max_x'], bb2['max_x'])
    y_bottom = min(bb1['max_y'], bb2['max_y'])

    if x_right < x_left or y_bottom < y_top:
        return None

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = float((bb1['max_x'] - bb1['min_x']) * (bb1['max_y'] - bb1['min_y']) + (bb2['max_x'] - bb2['min_x']) * (bb2['max_y'] - bb2['min_y']) - intersection_area)

    iou = intersection_area / union_area
    assert 1.0 >= iou >= 0.0
    return iou


with open(r'E:\ICDAR2019_test\final_result_0.7.json') as f:
    prediction = json.load(f)

final_result = dict()
for filename in prediction.keys():
    cells = []
    for table in prediction[filename]:
        for cell1 in table['cells']:
            skip = False
            for cell2 in cells:
                if calculate_iou(cell1, cell2) is not None:
                    if cell1['confidence'] > cell2['confidence']:
                        cell2 = cell1
                    skip = True
                    break
            if not skip:
                cells.append(cell1)

    final_result[filename] = cells

with open(r'E:\ICDAR2019_test\final_result_0.7_nonoverlap.json', 'w') as f:
    json.dump(final_result, f)
