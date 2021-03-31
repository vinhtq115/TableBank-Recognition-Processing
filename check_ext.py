import os

IMAGE_DIR = r'E:\TableBank-Recognition\Recognition\images'

extensions = set()

for _, _, files in os.walk(IMAGE_DIR):
    for file in files:
        extensions.add(file.split('.')[-1])
    print(extensions)
