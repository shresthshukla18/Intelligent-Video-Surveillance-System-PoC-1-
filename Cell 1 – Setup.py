#cell 1

#=========================================
#DRIVE CONTECTIVITY
#=========================================

from google.colab import drive
drive.mount('/content/drive')

#=========================================
#PROJECT FOLDER
#=========================================

import os

BASE_PATH = "/content/drive/MyDrive/cv_project"

VIDEO_FOLDER = f"{BASE_PATH}/videos"
OUTPUT_FOLDER = f"{BASE_PATH}/output"

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Project folders created.")

#=========================================
#PIP INSTALLATION
#=========================================

!pip install ultralytics supervision opencv-python

#========================================
#IMPORT
#========================================

from ultralytics import YOLO
import supervision as sv
import cv2

print("Libraries loaded successfully")
