import craft_wrapper
import os
import cv2
import numpy as np

CRAFT = craft_wrapper.CRAFT_pytorch()

PATH_TO_IMAGE_FOLDER = r'C:\Users\starc\PycharmProjects\TableBank-Recognition-Processing\images'  # Path to original images
image_name = r'SM08-Harris-Street-Village-FES-2012_7'
image_path = os.path.join(PATH_TO_IMAGE_FOLDER, image_name + '.png')
img = cv2.imread(image_path)

# Run the detector
bboxes, polys, heatmap = CRAFT.detect_text(img)

print(bboxes)
print('---------')
print(polys)
print('---------')
print(heatmap)
print('---------')


def show_bounding_boxes(img, bboxes):
    img = np.array(img)
    for i, box in enumerate(bboxes):
        poly = np.array(box).astype(np.int32).reshape((-1))

        poly = poly.reshape(-1, 2)
        cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)

    return img

# view the image with bounding boxes
img_boxed = show_bounding_boxes(img, bboxes)
cv2.imshow('fig', img_boxed)
cv2.waitKey()

# view detection heatmap
cv2.imshow('fig', heatmap)
cv2.waitKey()