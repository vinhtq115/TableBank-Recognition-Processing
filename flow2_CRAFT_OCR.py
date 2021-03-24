import os
import cv2
import craft_wrapper
import numpy as np
import time
from tqdm import tqdm
from utils import *
from lxml import etree

TXT_FLOW1 = r'E:\TableBank-Recognition\Recognition\flow1.txt'
PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\Recognition\images'
PATH_TO_ORIGINAL_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\annotations_original'
PATH_TO_DESTINATION_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\annotations'
CRAFT = None

file_list = open(r'E:\TableBank-Recognition\Recognition\flow2_ocr.txt', 'w')


def check_horizontal_line(img_segment_mid, background_pixel_value):
    mid_pixel_values = np.unique(img_segment_mid)
    if len(mid_pixel_values) == 1:
        if mid_pixel_values[0] != background_pixel_value:
            return True  # Have only pixel value that different from background => Has line
    else:
        full_line = False
        for line in img_segment_mid:
            values, counts = np.unique(line, return_counts=True)
            sum_other_than_background_pixel = 0
            sum_background_pixel = 0
            for i in range(len(values)):
                if values[i] != background_pixel_value:
                    sum_other_than_background_pixel += counts[i]
                else:
                    sum_background_pixel += counts[i]
            if sum_background_pixel / 4 < sum_other_than_background_pixel:
                full_line = True
                break
        if full_line:
            return True
        return False


def flow2(file):
    # Setup paths
    image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file + '.png')
    original_annotation_xml = os.path.join(PATH_TO_ORIGINAL_ANNOTATIONS, file + '.txt')
    destination_annotation_xml = os.path.join(PATH_TO_DESTINATION_ANNOTATIONS, file + '.xml')

    xml = open(original_annotation_xml).read()
    # Skip advanced table for later
    if advanced_table_check(xml):
        return

    img = cv2.imread(image_path, 0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # Detect text
    bboxes, polys, heatmap = CRAFT.detect_text(img)
    # Clean bounding boxes
    cleaned_bboxes = []
    for i, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))
        min_x = int(min(poly[0], poly[6]))
        max_x = int(max(poly[2], poly[4]))
        min_y = int(min(poly[1], poly[3]))
        max_y = int(max(poly[5], poly[7]))
        cleaned_bboxes.append([min_x, min_y, max_x, max_y])

    for bbox1 in cleaned_bboxes:
        for bbox2 in cleaned_bboxes:
            if bbox1 is bbox2:  # If bbox1 is bbox2 (and vice versa?), skip
                continue
            if bbox2[3] < bbox1[1] or bbox1[3] < bbox2[1]:  # bbox1 and bbox2 not vertically intersect
                #vertically_intersect = False
                height1 = bbox1[3] - bbox1[1]
                height2 = bbox2[3] - bbox2[1]
                height = min(height1, height2)
                if not ((bbox2[3] < bbox1[1] and bbox1[1] - bbox2[3] <= height // 4) or (
                        bbox1[3] < bbox2[1] and bbox2[1] - bbox1[3] <= height // 4)):
                    continue  # Make sure their vertical distance is not too far
                if bbox2[2] < bbox1[0] or bbox1[2] < bbox2[0]:
                    continue  # Skip those that are also not horizontally aligned. Maybe fixed later?

                img_segment_1 = img[bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]]
                img_segment_2 = img[bbox2[1]: bbox2[3], bbox2[0]: bbox2[2]]
                hist1 = cv2.calcHist([img_segment_1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([img_segment_2], [0], None, [256], [0, 256])
                background_pixel_1 = np.argmax(hist1)
                background_pixel_2 = np.argmax(hist2)
                if background_pixel_1 != background_pixel_2:
                    continue  # Skip if different background pixel values
                # Check if any line in between
                mid_area_min_x = max(bbox1[0], bbox2[0])
                mid_area_max_x = min(bbox1[2], bbox2[2])
                if bbox2[3] < bbox1[1]:
                    mid_area_min_y = bbox2[3]
                    mid_area_max_y = bbox1[1]
                else:
                    mid_area_min_y = bbox1[3]
                    mid_area_max_y = bbox2[1]
                img_segment_mid = img[mid_area_min_y: mid_area_max_y, mid_area_min_x: mid_area_max_x]
                # Check for horizontal lines
                if check_horizontal_line(img_segment_mid, background_pixel_1):
                    continue
            if bbox2[2] < bbox1[0] or bbox1[2] < bbox2[0]:
                # If these bboxes are not horizontally intersect but stay very near to each other, merge them
                if bbox2[3] < bbox1[1] or bbox1[3] < bbox2[1]:
                    continue  # Skip those that are also not vertically aligned. Maybe fixed later?
                height1 = bbox1[3] - bbox1[1]
                height2 = bbox2[3] - bbox2[1]
                height = min(height1, height2)
                if not ((bbox2[2] < bbox1[0] and bbox1[0] - bbox2[2] <= height // 4) or (
                        bbox1[2] < bbox2[0] and bbox2[0] - bbox1[2] <= height // 4)):
                    continue  # Make sure their horizontal distance is not too far
                img_segment_1 = img[bbox1[1]: bbox1[3], bbox1[0]: bbox1[2]]
                img_segment_2 = img[bbox2[1]: bbox2[3], bbox2[0]: bbox2[2]]
                hist1 = cv2.calcHist([img_segment_1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([img_segment_2], [0], None, [256], [0, 256])
                background_pixel_1 = np.argmax(hist1)
                background_pixel_2 = np.argmax(hist2)
                if background_pixel_1 != background_pixel_2:
                    continue  # Skip if different background pixel values
                # Check if any line in between
                mid_area_min_y = max(bbox1[1], bbox2[1])
                mid_area_max_y = min(bbox1[3], bbox2[3])
                if bbox2[2] < bbox1[0]:
                    mid_area_min_x = bbox2[2]
                    mid_area_max_x = bbox1[0]
                else:
                    mid_area_min_x = bbox1[2]
                    mid_area_max_x = bbox2[0]
                img_segment_mid = np.transpose(img[mid_area_min_y: mid_area_max_y, mid_area_min_x: mid_area_max_x])
                # Check vertical lines
                if check_horizontal_line(img_segment_mid, background_pixel_1):
                    continue
            bbox1[0] = min(bbox1[0], bbox2[0])
            bbox1[1] = min(bbox1[1], bbox2[1])
            bbox1[2] = max(bbox1[2], bbox2[2])
            bbox1[3] = max(bbox1[3], bbox2[3])
            cleaned_bboxes.remove(bbox2)
    bboxes_converted = []
    for bbox in cleaned_bboxes:
        bboxes_converted.append({'min_x': bbox[0], 'min_y': bbox[1], 'max_x': bbox[2], 'max_y': bbox[3]})

    # Read original XML file
    total_cells, total_cells_non_empty, rows = count_cells(original_annotation_xml)
    max_columns = -1
    max_visible_columns = -1
    for row in rows:
        if len(row) > max_columns:
            max_columns = len(row)
        if row.count('tdy') > max_visible_columns:
            max_visible_columns = row.count('tdy')

    rows_idx = [-1] * img.shape[0]
    row_idx = 0
    for i in range(img.shape[0]):
        for bbox in cleaned_bboxes:
            if bbox[1] <= i <= bbox[3]:
                rows_idx[i] = row_idx
                break
        if 0 < i <= img.shape[0] and rows_idx[i - 1] == row_idx != rows_idx[i]:
            row_idx += 1

    bboxes_clone = bboxes_converted.copy()
    for bbox in bboxes_clone:
        bbox['belong_to_row'] = rows_idx[bbox['min_y']]

    rows_2 = {}
    for bbox in bboxes_clone:
        if rows_2.get(bbox['belong_to_row']) is None:
            rows_2[bbox['belong_to_row']] = []
        rows_2[bbox['belong_to_row']].append(bbox)

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
    try:
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
    except:
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

        for bb in bboxes_converted:
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
            _xmin.text = str(bb['min_x'])

            # ymin
            _ymin = etree.SubElement(bndbox, 'ymin')
            _ymin.text = str(bb['min_y'])

            # xmax
            _xmax = etree.SubElement(bndbox, 'xmax')
            _xmax.text = str(bb['max_x'])

            # ymax
            _ymax = etree.SubElement(bndbox, 'ymax')
            _ymax.text = str(bb['max_y'])

        et = etree.ElementTree(root)
        et.write(destination_annotation_xml, pretty_print=True)
        file_list.write(file + '\n')
    else:
        return


if __name__ == '__main__':
    with open(TXT_FLOW1) as f:
        finised_files = f.readlines()
        finised_files = [x.strip() for x in finised_files]

        remaining_files = []
        for root, dirs, files in os.walk(PATH_TO_IMAGE_FOLDER):
            remaining_files = [get_file_name(x) for x in files if get_file_name(x) not in finised_files]
            # Initialize CRAFT
            start = time.time()
            CRAFT = craft_wrapper.CRAFT_pytorch()
            for file in tqdm(remaining_files):
                flow2(file)
            file_list.close()
