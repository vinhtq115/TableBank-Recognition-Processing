import cv2
import numpy as np
from lxml import etree

# Read image annotation file

# Parse XML
parser = etree.XMLParser(ns_clean=True,
                         remove_blank_text=True)  # Set up XML parser that remove namespace and blanks
xml_tree = etree.parse(r'E:\TableBank-Recognition\sampled\annotations\%C2%A0Richardson%2016_1.xml', parser)
tabular = xml_tree.getroot()  # <tabular>

# Read and store rows and columns
rows = []  # Store rows and columns
for i in tabular:  # Traverse <thead> and <tbody>
    for row in i:  # Traverse <tr>
        row_s = []
        for cell in row:
            if cell.tag == 'tdy':
                row_s.append('tdy')
            elif cell.tag == 'tdn':
                row_s.append('tdn')
        rows.append(row_s)

total = 0
total_cell_with_data = 0
for idx, row in enumerate(rows, 1):
    total += len(row)
    for cell in row:
        if cell == 'tdy':
            total_cell_with_data += 1

# Read image as grayscale
img = cv2.imread(r'E:\TableBank-Recognition\sampled\images\%C2%A0Richardson%2016_1.png', 0)

# Threshold the image. THRESH_OTSU for optimal value since it can threshold grey text correctly
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Invert the image
inverted_bin_img = 255 - binary

# cv2.imshow('1. Original', img)
# cv2.imshow('2. Binary', binary)
cv2.imshow('3. Inverted', inverted_bin_img)

# Detect table lines

# Set kernel size
# TODO: change divisor number based on number of rows, columns
horizontal_kernel_length = img.shape[1] // 50
vertical_kernel_length = img.shape[0] // 12

# Horizontal kernel for detecting horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_length, 1))

# Vertical kernel for detecting vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_length))

# Detect vertical lines
image_1 = cv2.erode(inverted_bin_img, vertical_kernel, iterations=3)
vertical_lines = cv2.dilate(image_1, vertical_kernel, iterations=3)
# cv2.imshow('4. Erode vert.', image_1)
cv2.imshow('4. Detected vertical lines', vertical_lines)

# Detect horizontal lines
image_2 = cv2.erode(inverted_bin_img, horizontal_kernel, iterations=3)
horizontal_lines = cv2.dilate(image_2, horizontal_kernel, iterations=3)
# cv2.imshow('5. Erode hor.', image_2)
cv2.imshow('5. Detected horizontal lines', horizontal_lines)

img_vh = cv2.bitwise_or(vertical_lines, horizontal_lines)
cv2.imshow('Vert + Hor', img_vh)

