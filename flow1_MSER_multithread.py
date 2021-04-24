from tqdm.contrib.concurrent import process_map
import os
import cv2
import numpy as np
from utils import *
from lxml import etree

PATH_TO_IMAGE_FOLDER = r'images'
PATH_TO_ORIGINAL_ANNOTATIONS = r'original_annotations'
#PATH_TO_DESTINATION_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\flow1'


def flow1(file):
    name = get_file_name(file)

    original_annotation_xml = os.path.join(PATH_TO_ORIGINAL_ANNOTATIONS, name + '.xml')
    image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file)
    #destination_annotation_xml = os.path.join(PATH_TO_DESTINATION_ANNOTATIONS, name + '.xml')

    xml = open(original_annotation_xml).read()
    # Skip advanced table for later
    if advanced_table_check(xml):
        return

    total_cells, total_cells_non_empty, rows = count_cells(original_annotation_xml)
    max_columns = -1
    max_visible_columns = -1
    for row in rows:
        if len(row) > max_columns:
            max_columns = len(row)
        if row.count('tdy') > max_visible_columns:
            max_visible_columns = row.count('tdy')

    # Read and threshold image
    img = cv2.imread(image_path, 0)  # Read image as grayscale
    _, binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)
    #_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the image
    inverted_bin_img = 255 - binary
    draw_border(inverted_bin_img)

    # Set kernel size for erosion/dilation.
    horizontal_kernel_length = img.shape[1] // max_columns // 4
    vertical_kernel_length = img.shape[0] // len(rows) // 4 * 3

    if horizontal_kernel_length < 1 or vertical_kernel_length < 1:
        return

    # Horizontal kernel for detecting horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))

    # Vertical kernel for detecting vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))

    # Detect vertical lines
    image_1 = cv2.erode(inverted_bin_img, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, vertical_kernel, iterations=3)

    # Detect horizontal lines
    image_2 = cv2.erode(inverted_bin_img, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, horizontal_kernel, iterations=3)

    # Generate lines mask
    lines_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)

    # Remove lines from threshold image (also remove white pixel noise)
    inverted_bin_img_removed_lines = inverted_bin_img - lines_mask
    _kernel = np.ones((1, 1), np.uint8)
    inverted_bin_img_removed_lines = cv2.morphologyEx(inverted_bin_img_removed_lines, cv2.MORPH_OPEN, _kernel)

    # Manually detect table cells using MSER
    # Initialize MSER
    mser = cv2.MSER_create(_min_area=4)

    # Detect regions and draw bounding boxes
    img2 = 255 - inverted_bin_img_removed_lines
    _, bboxes = mser.detectRegions(img2)

    empty_img2 = np.full_like(img2, 255)
    # Draw bounding boxes
    for bbox in bboxes:
        cv2.rectangle(empty_img2, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 0), -1)

    # Get bounding box height that appear the most.
    # Idea: merge letters and texts on the same line using that height.
    heights = {}
    for bbox in bboxes:
        if heights.get(str(bbox[3])) is None:
            heights[str(bbox[3])] = 1
        else:
            heights[str(bbox[3])] += 1

    most_height = 0
    most_height_count = 0

    for key, value in heights.items():
        if value > most_height_count:
            most_height = key
            most_height_count = value
    most_height = int(int(most_height) * 1.3)

    if most_height < 1:
        return

    reversed_img2 = ~empty_img2
    test_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (most_height, 1))
    img3 = cv2.morphologyEx(reversed_img2, cv2.MORPH_CLOSE, test_kernel)

    # Find rows
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img3, connectivity=4)

    bboxes_converted = []
    for label in range(1, nb_components):
        min_x = stats[label][0]
        min_y = stats[label][1]
        max_x = stats[label][0] + stats[label][2]
        max_y = stats[label][1] + stats[label][3]
        if (max_x - min_x) * (max_y - min_y) > 4:
            bb = {'id': label, 'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}
            bboxes_converted.append(bb)

    # Assign bounding boxes to corresponding row
    rows_idx = [-1] * img.shape[0]
    row_idx = 0
    for i in range(img.shape[0]):
        for bbox in bboxes:
            if bbox[1] <= i <= bbox[1] + bbox[3]:
                rows_idx[i] = row_idx
                break
        if 0 < i <= img.shape[0] and rows_idx[i - 1] == row_idx != rows_idx[i]:
            row_idx += 1

    bboxes_clone = bboxes_converted.copy()
    for bbox in bboxes_clone:
        bbox['belong_to_row'] = rows_idx[bbox['min_y']]

    # Convert to dict of list of dict
    rows_2 = {}
    for bbox in bboxes_clone:
        if rows_2.get(bbox['belong_to_row']) is None:
            rows_2[bbox['belong_to_row']] = []
        rows_2[bbox['belong_to_row']].append(bbox)

    # Assign bounding boxes to corresponding column
    sorted_rows = {}
    max_detected_column = -1
    for i, _row in rows_2.items():
        sorted_rows[i] = sorted(_row, key=lambda x: x['min_x'])
        for j in range(len(_row)):
            sorted_rows[i][j]['belong_to_column'] = j
            if j > max_detected_column:
                max_detected_column = j

    # Check if number of rows/columns in annotation file and in detected rows/columns match
    matched = True
    if len(rows) == row_idx and max_visible_columns == max_detected_column + 1:
        rows_mapped = []
        for i in range(len(rows)):
            rows_temp = []
            visible_col_count = 0
            for j in range(len(rows[i])):
                if rows[i][j] == 'tdn':
                    rows_temp.append({'tdn': {}})
                elif visible_col_count < len(sorted_rows[i]):
                    rows_temp.append({'tdy': sorted_rows[i][visible_col_count]})
                    visible_col_count += 1
                else:
                    matched = False
                    break
            if visible_col_count != rows[i].count('tdy') or not matched:
                matched = False
                break
            rows_mapped.append(rows_temp)
    else:
        matched = False

    if matched:
        return True
    else:
        return False


def flow1e(file):
    name = get_file_name(file)

    original_annotation_xml = os.path.join(PATH_TO_ORIGINAL_ANNOTATIONS, name + '.xml')
    image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file)
    #destination_annotation_xml = os.path.join(PATH_TO_DESTINATION_ANNOTATIONS, name + '.xml')

    xml = open(original_annotation_xml).read()
    # Skip advanced table for later
    if advanced_table_check(xml):
        return

    total_cells, total_cells_non_empty, rows = count_cells(original_annotation_xml)
    max_columns = -1
    max_visible_columns = -1
    for row in rows:
        if len(row) > max_columns:
            max_columns = len(row)
        if row.count('tdy') > max_visible_columns:
            max_visible_columns = row.count('tdy')

    # Read and threshold image
    img = cv2.imread(image_path, 0)  # Read image as grayscale
    #_, binary = cv2.threshold(img, 225, 255, cv2.THRESH_BINARY)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the image
    inverted_bin_img = 255 - binary
    draw_border(inverted_bin_img)

    # Set kernel size for erosion/dilation.
    horizontal_kernel_length = img.shape[1] // max_columns // 4
    vertical_kernel_length = img.shape[0] // len(rows) // 4 * 3

    if horizontal_kernel_length < 1 or vertical_kernel_length < 1:
        return

    # Horizontal kernel for detecting horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))

    # Vertical kernel for detecting vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))

    # Detect vertical lines
    image_1 = cv2.erode(inverted_bin_img, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, vertical_kernel, iterations=3)

    # Detect horizontal lines
    image_2 = cv2.erode(inverted_bin_img, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, horizontal_kernel, iterations=3)

    # Generate lines mask
    lines_mask = cv2.bitwise_or(vertical_lines, horizontal_lines)

    # Remove lines from threshold image (also remove white pixel noise)
    inverted_bin_img_removed_lines = inverted_bin_img - lines_mask
    _kernel = np.ones((1, 1), np.uint8)
    inverted_bin_img_removed_lines = cv2.morphologyEx(inverted_bin_img_removed_lines, cv2.MORPH_OPEN, _kernel)

    # Manually detect table cells using MSER
    # Initialize MSER
    mser = cv2.MSER_create(_min_area=4)

    # Detect regions and draw bounding boxes
    img2 = 255 - inverted_bin_img_removed_lines
    _, bboxes = mser.detectRegions(img2)

    empty_img2 = np.full_like(img2, 255)
    # Draw bounding boxes
    for bbox in bboxes:
        cv2.rectangle(empty_img2, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 0), -1)

    # Get bounding box height that appear the most.
    # Idea: merge letters and texts on the same line using that height.
    heights = {}
    for bbox in bboxes:
        if heights.get(str(bbox[3])) is None:
            heights[str(bbox[3])] = 1
        else:
            heights[str(bbox[3])] += 1

    most_height = 0
    most_height_count = 0

    for key, value in heights.items():
        if value > most_height_count:
            most_height = key
            most_height_count = value
    most_height = int(int(most_height) * 1.3)

    if most_height < 1:
        return

    reversed_img2 = ~empty_img2
    test_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (most_height, 1))
    img3 = cv2.morphologyEx(reversed_img2, cv2.MORPH_CLOSE, test_kernel)

    # Find rows
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img3, connectivity=4)

    bboxes_converted = []
    for label in range(1, nb_components):
        min_x = stats[label][0]
        min_y = stats[label][1]
        max_x = stats[label][0] + stats[label][2]
        max_y = stats[label][1] + stats[label][3]
        if (max_x - min_x) * (max_y - min_y) > 4:
            bb = {'id': label, 'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}
            bboxes_converted.append(bb)

    # Assign bounding boxes to corresponding row
    rows_idx = [-1] * img.shape[0]
    row_idx = 0
    for i in range(img.shape[0]):
        for bbox in bboxes:
            if bbox[1] <= i <= bbox[1] + bbox[3]:
                rows_idx[i] = row_idx
                break
        if 0 < i <= img.shape[0] and rows_idx[i - 1] == row_idx != rows_idx[i]:
            row_idx += 1

    bboxes_clone = bboxes_converted.copy()
    for bbox in bboxes_clone:
        bbox['belong_to_row'] = rows_idx[bbox['min_y']]

    # Convert to dict of list of dict
    rows_2 = {}
    for bbox in bboxes_clone:
        if rows_2.get(bbox['belong_to_row']) is None:
            rows_2[bbox['belong_to_row']] = []
        rows_2[bbox['belong_to_row']].append(bbox)

    # Assign bounding boxes to corresponding column
    sorted_rows = {}
    max_detected_column = -1
    for i, _row in rows_2.items():
        sorted_rows[i] = sorted(_row, key=lambda x: x['min_x'])
        for j in range(len(_row)):
            sorted_rows[i][j]['belong_to_column'] = j
            if j > max_detected_column:
                max_detected_column = j

    # Check if number of rows/columns in annotation file and in detected rows/columns match
    matched = True
    if len(rows) == row_idx and max_visible_columns == max_detected_column + 1:
        rows_mapped = []
        for i in range(len(rows)):
            rows_temp = []
            visible_col_count = 0
            for j in range(len(rows[i])):
                if rows[i][j] == 'tdn':
                    rows_temp.append({'tdn': {}})
                elif visible_col_count < len(sorted_rows[i]):
                    rows_temp.append({'tdy': sorted_rows[i][visible_col_count]})
                    visible_col_count += 1
                else:
                    matched = False
                    break
            if visible_col_count != rows[i].count('tdy') or not matched:
                matched = False
                break
            rows_mapped.append(rows_temp)
    else:
        matched = False

    if matched:
        return True
    else:
        return False

if __name__ == '__main__':
    for root, dirs, files in os.walk(PATH_TO_IMAGE_FOLDER):
        result = process_map(flow1, files, max_workers=12, chunksize=15)
        result2 = process_map(flow1e, files, max_workers=12, chunksize=15)  # Otsu
        for i in range(len(files)):
            if result[i] is True and result2[i] is False:
                print(files[i])
