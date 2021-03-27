import os
import multiprocessing
import cv2
import numpy as np
import time
from utils import *
from lxml import etree

PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\Recognition\images'
PATH_TO_ORIGINAL_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\annotations_original'
PATH_TO_DESTINATION_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\flow3'


def trim_bbox(img, bbox):
    top = bbox[0]
    left = bbox[1]
    bottom = bbox[2]
    right = bbox[3]
    if len(np.unique(img[top : bottom + 1, left : right + 1])) == 1:
        return None  # Empty cells. No need to trim. Just return None
    while top < bottom:
        if len(np.unique(img[top, left: right + 1])) == 1:
            top += 1
        else:
            break
    while bottom > top:
        if len(np.unique(img[bottom, left: right + 1])) == 1:
            bottom -= 1
        else:
            break
    while left < right:
        if len(np.unique(img[top : bottom + 1, left])) == 1:
            left += 1
        else:
            break
    while right > left:
        if len(np.unique(img[top : bottom + 1, right])) == 1:
            right -= 1
        else:
            break
    return [top, left, bottom, right]


def flow3(file):
    file = get_file_name(file)
    # Setup paths
    image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file + '.png')
    original_annotation_xml = os.path.join(PATH_TO_ORIGINAL_ANNOTATIONS, file + '.txt')
    destination_annotation_xml = os.path.join(PATH_TO_DESTINATION_ANNOTATIONS, file + '.xml')

    xml = open(original_annotation_xml).read()
    # Skip advanced table for later
    if advanced_table_check(xml):
        return

    # Read original annotations
    total_cells, total_cells_non_empty, rows_ann = count_cells(original_annotation_xml)
    max_columns = -1
    max_visible_columns = -1
    for row in rows_ann:
        if len(row) > max_columns:
            max_columns = len(row)
        if row.count('tdy') > max_visible_columns:
            max_visible_columns = row.count('tdy')

    # Read and threshold image
    img = cv2.imread(image_path, 0)  # Read image as grayscale
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Invert the image
    inverted_bin_img = 255 - binary
    draw_border(inverted_bin_img)

    # Set kernel size for erosion/dilation.
    horizontal_kernel_length = img.shape[1] // max_columns // 4
    vertical_kernel_length = img.shape[0] // len(rows_ann) // 4 * 3

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

    rows_coordinates = []
    rows_vertical_projection = [False] * len(horizontal_lines)  # False means black, True means white
    for i in range(len(horizontal_lines)):
        if np.unique(horizontal_lines[i])[0] == 255:
            rows_vertical_projection[i] = True

    row_idx = 0
    while row_idx < len(horizontal_lines):
        quit = False
        while rows_vertical_projection[row_idx] == True:
            row_idx += 1
            if row_idx == len(horizontal_lines) - 1:
                quit = True
                break
        if quit:
            break
        row_start = row_idx
        while rows_vertical_projection[row_idx] == False:
            row_idx += 1
        row_end = row_idx - 1
        rows_coordinates.append([row_start, row_end])
        if row_idx >= len(horizontal_lines) - 1:
            break

    rows_coordinates = [row_coordinates for row_coordinates in rows_coordinates if row_coordinates[1] - row_coordinates[0] > 4]

    cols_coordinates = []
    cols_horizontal_projection = [False] * len(vertical_lines[0])  # False means black, True means white
    for i in range(len(vertical_lines[0])):
        if np.unique(vertical_lines[:, i])[0] == 255:
            cols_horizontal_projection[i] = True

    col_idx = 0
    while col_idx < len(vertical_lines[0]):
        quit = False
        while cols_horizontal_projection[col_idx] == True:
            col_idx += 1
            if col_idx == len(vertical_lines[0]) - 1:
                quit = True
                break
        if quit:
            break
        col_start = col_idx
        while cols_horizontal_projection[col_idx] == False:
            col_idx += 1
        col_end = col_idx - 1
        cols_coordinates.append([col_start, col_end])
        if col_idx >= len(vertical_lines[0]) - 1:
            break

    cols_coordinates = [col_coordinates for col_coordinates in cols_coordinates if col_coordinates[1] - col_coordinates[0] > 4]

    cells = []
    for row_idx, row in enumerate(rows_coordinates):
        row_temp = []
        for col in cols_coordinates:
            row_temp.append([row[0], col[0], row[1], col[1]])
        cells.append(row_temp)

    new_cells = []
    for i in range(len(cells)):
        new_row = []
        for j in range(len(cells[i])):
            new_bbox = trim_bbox(inverted_bin_img, cells[i][j])
            if new_bbox is not None:
                new_row.append(new_bbox)
        new_cells.append(new_row)

    # Check if number of rows/columns in annotation file and in detected rows/columns match
    matched = True
    if len(rows_ann) == len(new_cells):  # Check if numbers of rows are equal
        for row in range(len(rows_ann)):
            # Count visible cell in current row of annotation
            visible_col_count = 0
            for cell in rows_ann[row]:
                if cell == 'tdy':
                    visible_col_count += 1
            if visible_col_count != len(new_cells[row]):
                matched = False
                break
    else:
        matched = False

    if matched:
        # Write to XML
        root = etree.Element('annotation')
        # folder tag
        folder = etree.SubElement(root, 'folder')
        folder.text = 'images'
        # filename tag
        filename = etree.SubElement(root, 'filename')
        filename.text = file
        # path tag
        path = etree.SubElement(root, 'path')
        path.text = PATH_TO_IMAGE_FOLDER
        # source tag
        source = etree.SubElement(root, 'source')
        # database tag
        database = etree.SubElement(source, 'database')
        database.text = 'Unknown'
        # size tag
        size = etree.SubElement(root, 'size')
        _image = cv2.imread(image_path)
        width = etree.SubElement(size, 'width')
        width.text = str(_image.shape[1])
        height = etree.SubElement(size, 'height')
        height.text = str(_image.shape[0])
        depth = etree.SubElement(size, 'depth')
        depth.text = str(_image.shape[2])
        # segmented tag
        segmented = etree.SubElement(root, 'segmented')
        segmented.text = str(0)

        for row in new_cells:
            for cell in row:
                object = etree.SubElement(root, 'object')
                # name
                name = etree.SubElement(object, 'name')
                name.text = 'table_cell'
                # pose
                pose = etree.SubElement(object, 'pose')
                pose.text = 'Unspecified'
                # truncated
                truncated = etree.SubElement(object, 'truncated')
                truncated.text = '0'
                # difficult
                difficult = etree.SubElement(object, 'difficult')
                difficult.text = '0'
                # bndbox
                bndbox = etree.SubElement(object, 'bndbox')
                # xmin
                _xmin = etree.SubElement(bndbox, 'xmin')
                _xmin.text = str(cell[0])

                # ymin
                _ymin = etree.SubElement(bndbox, 'ymin')
                _ymin.text = str(cell[1])

                # xmax
                _xmax = etree.SubElement(bndbox, 'xmax')
                _xmax.text = str(cell[2])

                # ymax
                _ymax = etree.SubElement(bndbox, 'ymax')
                _ymax.text = str(cell[3])
        et = etree.ElementTree(root)
        et.write(destination_annotation_xml, pretty_print=True)


if __name__ == '__main__':
    a_pool = multiprocessing.Pool()
    total = 0
    for root, dirs, files in os.walk(PATH_TO_IMAGE_FOLDER):
        start = time.time()
        a_pool.map(flow3, files)
        end = time.time()
        total = end - start
    print('Time taken: ' + str(total) + ' (s)')

    # for root, dirs, files in os.walk(PATH_TO_IMAGE_FOLDER):
    #     for file in tqdm(files):
    #         try:
    #             flow3(get_file_name(file))
    #         except:
    #             print(get_file_name(file))
    # print('Advanced files: ' + str(advance_counter))
    # print('Correct files: ' + str(correct_counter))
