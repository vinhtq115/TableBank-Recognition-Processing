import cv2
import os
from lxml import etree
from shutil import copyfile
from utils import get_file_name

PATH_TO_ANNOTATION_FOLDER = r'E:\TableBank-Recognition\final_recognition_data\annotations'
PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\final_recognition_data\images'

# Check if both folders have matching files
ann_files = []
for _, _, files in os.walk(PATH_TO_ANNOTATION_FOLDER):
    ann_files = files
img_files = []
for _, _, files in os.walk(PATH_TO_IMAGE_FOLDER):
    img_files = files

assert len(ann_files) == len(img_files)
ann_files_no_ext = [get_file_name(ann_file) for ann_file in ann_files]
img_files_no_ext = [get_file_name(img_file) for img_file in img_files]
for i in range(len(ann_files)):
    assert ann_files_no_ext[i] == img_files_no_ext[i]
del ann_files_no_ext, img_files_no_ext

PATH_TO_ANNOTATION_FOLDER_FLOW_1 = r'E:\TableBank-Recognition\Recognition\flows\flow1'
PATH_TO_ANNOTATION_FOLDER_FLOW_2_1 = r'E:\TableBank-Recognition\Recognition\flows\flow2_1'
PATH_TO_ANNOTATION_FOLDER_FLOW_2_2 = r'E:\TableBank-Recognition\Recognition\flows\flow2_2'
PATH_TO_ANNOTATION_FOLDER_FLOW_3 = r'E:\TableBank-Recognition\Recognition\flows\flow3'
PATH_TO_ANNOTATION_FOLDER_FLOW_4 = r'E:\TableBank-Recognition\Recognition\flows\flow4'


# Show each file + annotation
correct = 0


def draw_bbox(img_file, pascal_voc_xml_file):
    """
    Draw bounding boxes.
    :param img_file: Path to image file
    :param pascal_voc_xml_file: Path to annotation file
    :return: Image with bounding boxes drawn
    """
    img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
    xml_tree = etree.parse(pascal_voc_xml_file)
    root = xml_tree.getroot()
    res = img.copy()
    for child in root:
        if child.tag != 'object':
            continue
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        for child_element in child:
            if child_element.tag != 'bndbox':
                continue
            for unk_val in child_element:
                if unk_val.tag == 'xmin':
                    xmin = int(unk_val.text)
                elif unk_val.tag == 'ymin':
                    ymin = int(unk_val.text)
                elif unk_val.tag == 'xmax':
                    xmax = int(unk_val.text)
                elif unk_val.tag == 'ymax':
                    ymax = int(unk_val.text)
        cv2.rectangle(res, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return res


for i in range(len(ann_files)):
    # Setup path
    img_file = os.path.join(PATH_TO_IMAGE_FOLDER, img_files[i])
    ann_file = os.path.join(PATH_TO_ANNOTATION_FOLDER, ann_files[i])

    img_drawn = draw_bbox(img_file, ann_file)
    cv2.imshow('Result from annotations folder', img_drawn)
    choice = cv2.waitKey()

    if choice == 32:  # 32 = Space => Correct
        correct += 1
        continue

    # Try flow 1's annotation
    if os.path.isfile(os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_1, ann_files[i])):
        ann_file = os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_1, ann_files[i])
        img_drawn = draw_bbox(img_file, ann_file)
        cv2.imshow('Result from flow 1 folder', img_drawn)
        choice = cv2.waitKey()
        if choice == 32:  # 32 = Space => Correct
            correct += 1
            copyfile(ann_file, os.path.join(PATH_TO_ANNOTATION_FOLDER, ann_files[i]))
            continue

    # Try flow 2_1's annotation
    if os.path.isfile(os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_2_1, ann_files[i])):
        ann_file = os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_2_1, ann_files[i])
        img_drawn = draw_bbox(img_file, ann_file)
        cv2.imshow('Result from flow 2_1 folder', img_drawn)
        choice = cv2.waitKey()
        if choice == 32:  # 32 = Space => Correct
            correct += 1
            copyfile(ann_file, os.path.join(PATH_TO_ANNOTATION_FOLDER, ann_files[i]))
            continue

    # Try flow 2_2's annotation
    if os.path.isfile(os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_2_2, ann_files[i])):
        ann_file = os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_2_2, ann_files[i])
        img_drawn = draw_bbox(img_file, ann_file)
        cv2.imshow('Result from flow 2_2 folder', img_drawn)
        choice = cv2.waitKey()
        if choice == 32:  # 32 = Space => Correct
            correct += 1
            copyfile(ann_file, os.path.join(PATH_TO_ANNOTATION_FOLDER, ann_files[i]))
            continue

    # Try flow 3's annotation
    if os.path.isfile(os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_3, ann_files[i])):
        ann_file = os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_3, ann_files[i])
        img_drawn = draw_bbox(img_file, ann_file)
        cv2.imshow('Result from flow 3 folder', img_drawn)
        choice = cv2.waitKey()
        if choice == 32:  # 32 = Space => Correct
            correct += 1
            copyfile(ann_file, os.path.join(PATH_TO_ANNOTATION_FOLDER, ann_files[i]))
            continue

    # Try flow 4's annotation
    if os.path.isfile(os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_4, ann_files[i])):
        ann_file = os.path.join(PATH_TO_ANNOTATION_FOLDER_FLOW_4, ann_files[i])
        img_drawn = draw_bbox(img_file, ann_file)
        cv2.imshow('Result from flow 4 folder', img_drawn)
        choice = cv2.waitKey()
        if choice == 32:  # 32 = Space => Correct
            correct += 1
            copyfile(ann_file, os.path.join(PATH_TO_ANNOTATION_FOLDER, ann_files[i]))
            continue

    with open(r'E:\TableBank-Recognition\final_recognition_data\wrong_files.txt', 'a') as f:
        f.write(get_file_name(img_files[i]) + '\n')
