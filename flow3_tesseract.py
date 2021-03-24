from PIL import Image
import pytesseract

print(pytesseract.image_to_boxes(Image.open(r'images\%C2%A0Richardson%2016_1.png')))
