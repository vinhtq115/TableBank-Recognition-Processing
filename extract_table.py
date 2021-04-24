import os
import json
from lxml import etree
from tqdm import tqdm
from utils import get_file_name

ROOT_FOLDER = r'E:\ICDAR2019_test'
ANNOTATIONS_FOLDER = os.path.join(ROOT_FOLDER, 'annotations')

for _, _, files in os.walk(ANNOTATIONS_FOLDER):
    result = dict()
    for file in tqdm(files):
        filename = get_file_name(file)

        result[filename + '.jpg'] = []

        parser = etree.XMLParser(ns_clean=True, remove_blank_text=True)
        old_xml_tree = etree.parse(os.path.join(ANNOTATIONS_FOLDER, file), parser)
        annotation = old_xml_tree.getroot()

        for tag in annotation:
            if tag.tag == 'object':
                table = False
                for minitag in tag:
                    if minitag.tag == 'name' and minitag.text == 'table':
                        table = True
                if not table:
                    break
                min_x = min_y = max_x = max_y = None
                for minitag in tag:
                    if minitag.tag == 'bndbox':
                        for microtag in minitag:
                            if microtag.tag == 'xmin':
                                min_x = int(microtag.text)
                            elif microtag.tag == 'xmax':
                                max_x = int(microtag.text)
                            elif microtag.tag == 'ymin':
                                min_y = int(microtag.text)
                            elif microtag.tag == 'ymax':
                                max_y = int(microtag.text)
                temp_dic = {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y, 'confidence': 1.0, 'cells': []}
                result[filename + '.jpg'].append(temp_dic)
    with open(os.path.join(ROOT_FOLDER, 'table_gt.json'), 'w') as f:
        json.dump(result, f)
