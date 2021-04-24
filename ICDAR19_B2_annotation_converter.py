import os
import cv2
from lxml import etree
from tqdm import tqdm
from utils import get_file_name

ROOT_FOLDER = r'E:\ICDAR19_TRACK_A'
IMAGES_FOLDER = os.path.join(ROOT_FOLDER, 'images')
ANNOTATIONS_OLD_FOLDER = os.path.join(ROOT_FOLDER, 'annotations_original')
ANNOTATIONS_FOLDER = os.path.join(ROOT_FOLDER, 'annotations')

for _, _, files in os.walk(ANNOTATIONS_OLD_FOLDER):
    for file in tqdm(files):
        tables = []
        filename = get_file_name(file)

        img = cv2.imread(os.path.join(IMAGES_FOLDER, filename + '.jpg'))
        old_xml_file = os.path.join(ANNOTATIONS_OLD_FOLDER, filename + '.xml')
        destination_xml_file = os.path.join(ANNOTATIONS_FOLDER, filename + '.xml')

        parser = etree.XMLParser(ns_clean=True, remove_blank_text=True)
        old_xml_tree = etree.parse(old_xml_file, parser)
        document = old_xml_tree.getroot()

        for table in document:
            cur_table = {'min_x': None, 'min_y': None, 'max_x': None, 'max_y': None}
            for element in table:
                if element.tag == 'Coords':  # Table coordinate
                    points = element.attrib['points'].split(' ')
                    cur_table['min_x'] = int(points[0][:points[0].find(',')])
                    cur_table['min_y'] = int(points[0][points[0].find(',') + 1:])
                    cur_table['max_x'] = int(points[2][:points[2].find(',')])
                    cur_table['max_y'] = int(points[2][points[2].find(',') + 1:])
                # elif element.tag == 'cell':
                #     cell_chord = element[0]
                #     points = cell_chord.attrib['points'].split(' ')
                #     min_x = int(points[0][:points[0].find(',')])
                #     min_y = int(points[0][points[0].find(',') + 1:])
                #     max_x = int(points[2][:points[2].find(',')])
                #     max_y = int(points[2][points[2].find(',') + 1:])
                #     cur_table['cells'].append([min_x, min_y, max_x, max_y])
            tables.append(cur_table)

        # Save
        # Write to XML
        root = etree.Element('annotation')
        # folder tag
        folder = etree.SubElement(root, 'folder')
        folder.text = 'images'
        # filename tag
        filename = etree.SubElement(root, 'filename')
        filename.text = get_file_name(file) + '.jpg'
        # path tag
        path = etree.SubElement(root, 'path')
        path.text = IMAGES_FOLDER
        # source tag
        source = etree.SubElement(root, 'source')
        # database tag
        database = etree.SubElement(source, 'database')
        database.text = 'Unknown'
        # size tag
        size = etree.SubElement(root, 'size')
        width = etree.SubElement(size, 'width')
        width.text = str(img.shape[1])
        height = etree.SubElement(size, 'height')
        height.text = str(img.shape[0])
        depth = etree.SubElement(size, 'depth')
        depth.text = str(img.shape[2])
        # segmented tag
        segmented = etree.SubElement(root, 'segmented')
        segmented.text = str(0)

        for table in tables:
            # Save table
            object = etree.SubElement(root, 'object')
            # name
            name = etree.SubElement(object, 'name')
            name.text = 'table'
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
            _xmin.text = str(table['min_x'])

            # ymin
            _ymin = etree.SubElement(bndbox, 'ymin')
            _ymin.text = str(table['min_y'])

            # xmax
            _xmax = etree.SubElement(bndbox, 'xmax')
            _xmax.text = str(table['max_x'])

            # ymax
            _ymax = etree.SubElement(bndbox, 'ymax')
            _ymax.text = str(table['max_y'])
            # Save table cells
            # for cell in table['cells']:
            #     object = etree.SubElement(root, 'object')
            #     # name
            #     name = etree.SubElement(object, 'name')
            #     name.text = 'table_cell'
            #     # pose
            #     pose = etree.SubElement(object, 'pose')
            #     pose.text = 'Unspecified'
            #     # truncated
            #     truncated = etree.SubElement(object, 'truncated')
            #     truncated.text = '0'
            #     # difficult
            #     difficult = etree.SubElement(object, 'difficult')
            #     difficult.text = '0'
            #     # bndbox
            #     bndbox = etree.SubElement(object, 'bndbox')
            #     # xmin
            #     _xmin = etree.SubElement(bndbox, 'xmin')
            #     _xmin.text = str(cell[0])
            #
            #     # ymin
            #     _ymin = etree.SubElement(bndbox, 'ymin')
            #     _ymin.text = str(cell[1])
            #
            #     # xmax
            #     _xmax = etree.SubElement(bndbox, 'xmax')
            #     _xmax.text = str(cell[2])
            #
            #     # ymax
            #     _ymax = etree.SubElement(bndbox, 'ymax')
            #     _ymax.text = str(cell[3])
        et = etree.ElementTree(root)
        et.write(destination_xml_file, pretty_print=True)
