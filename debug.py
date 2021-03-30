import os
import cv2

PATH_TO_IMAGE_FOLDER = r'C:\Users\starc\PycharmProjects\TableBank-Recognition-Processing\images'
PATH_TO_ORIGINAL_ANNOTATIONS = r'C:\Users\starc\PycharmProjects\TableBank-Recognition-Processing\original_annotations'
file = '1491745306328040082_35'

image_path = os.path.join(PATH_TO_IMAGE_FOLDER, file + '.png')

img = cv2.imread(image_path, 0)
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

cv2.imshow('Binary', binary)
cv2.waitKey()
cv2.destroyAllWindows()

template = cv2.imread('missing_font_template.png', 0)

threshold = 0.8
res = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
new_image = binary.copy()
h, w = template.shape[:2]

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(min_val, max_val, min_loc, max_loc)
