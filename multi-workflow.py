# Import necessary packages
import os
import shutil
import cv2
import numpy as np
from utils import *
from lxml import etree

# Setup paths
PATH_TO_IMAGE_FOLDER = r'E:\TableBank-Recognition\Recognition\images'  # Path to original images
PATH_TO_ORIGINAL_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\annotations_original'  # Path to original annotations (converted to individual files)
PATH_TO_DESTINATION_ANNOTATIONS = r'E:\TableBank-Recognition\Recognition\annotations'  # Path to save new XML annotations

