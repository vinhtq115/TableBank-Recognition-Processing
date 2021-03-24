import os
import cv2
import numpy as np
from utils import *
from lxml import etree

PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\sampled\images'
PATH_TO_ORIGINAL_ANNOTATIONS = r'E:\TableBank-Recognition\sampled\annotations_original'
PATH_TO_DESTINATION_ANNOTATIONS = r'E:\TableBank-Recognition\sampled\annotations'
PATH_TO_IMAGE_BB_FOLDER = r'E:\TableBank-Recognition\sampled\images_bb'

advanced_file = []  # Containing files that either have annotation errors or spanning cells.

for root, dirs, files in os.walk(r'E:\TableBank-Recognition\sampled\images'):
    total = len(files)  # Total number of files
    advanced = 0
    correct = 0
    incorrect = 0

    for file in files:
        name = get_file_name(file)
        print(name)

        original_annotation_xml = os.path.join(PATH_TO_ORIGINAL_ANNOTATIONS, name + '.xml')
        image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file)
        save_image_path = os.path.join(PATH_TO_IMAGE_BB_FOLDER, file)
        destination_annotation_xml = os.path.join(PATH_TO_DESTINATION_ANNOTATIONS, name + '.xml')

        xml = open(original_annotation_xml).read()
        # Skip advanced table for later
        if advanced_table_check(xml):
            advanced_file.append(file)
            advanced += 1
            continue

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

        # Invert the image
        inverted_bin_img = 255 - binary
        draw_border(inverted_bin_img)
        # Fill dots that causing error
        # _to_be_filled = []
        # for _row in range(inverted_bin_img.shape[0]):
        #     for _col in range(inverted_bin_img.shape[1]):
        #         count = 0
        #         if inverted_bin_img[_row][_col] == 0:
        #             if _row - 1 >= 0 and inverted_bin_img[_row - 1][_col] == 255:
        #                 count += 1
        #             if _row + 1 < inverted_bin_img.shape[0] and inverted_bin_img[_row + 1][_col] == 255:
        #                 count += 1
        #             if _col - 1 >= 0 and inverted_bin_img[_row][_col - 1] == 255:
        #                 count += 1
        #             if _col + 1 < inverted_bin_img.shape[1] and inverted_bin_img[_row][_col + 1] == 255:
        #                 count += 1
        #         if count >= 2:
        #             _to_be_filled.append([_row, _col])
        # for it in _to_be_filled:
        #     inverted_bin_img[it[0]][it[1]] = 255

        # Set kernel size for erosion/dilation. TODO: change divisor number based on number of rows, columns
        horizontal_kernel_length = img.shape[1] // max_columns // 4
        vertical_kernel_length = img.shape[0] // len(rows) // 4 * 3

        if horizontal_kernel_length < 1 or vertical_kernel_length < 1:
            advanced_file.append(file)
            advanced += 1
            continue

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
        # Idea: merge letters and texts on the same line using that height. TODO: improve
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
            advanced_file.append(file)
            advanced += 1
            continue

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
            correct += 1
            # TODO: save xml for correct file
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
        else:
            incorrect += 1

        img_save = cv2.imread(image_path)
        color = (0, 0, 255) if not matched else (0, 255, 0)
        for bb in bboxes_converted:
            cv2.rectangle(img_save, (bb['min_x'], bb['min_y']), (bb['max_x'], bb['max_y']), color, 2)
        cv2.imwrite(os.path.join(PATH_TO_IMAGE_BB_FOLDER, file), img_save)

    print('--------------------')
    print('Number of files:', total)
    print('Number of advanced files (skipped):', advanced)
    print('Correct: {:.2%}'.format(correct/total))
    print('Incorrect: {:.2%}'.format(incorrect/total))
