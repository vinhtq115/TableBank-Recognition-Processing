import cv2

arr = cv2.imread('missing_font_template.png')

while True:
    cv2.imshow('fig', arr)
    a = cv2.waitKey()
    print(a)
