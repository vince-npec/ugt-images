import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re
import pandas as pd

# Define the mapping from file numbers to insect names
insect_names = {
    0: "Ulysses",
    1: "Admiral",
    2: "Scarab",
    3: "Ladybug",
    4: "Yellowjacket",
    5: "Flea",
    6: "Mosquito",
    7: "Stag beetle",
    8: "Cockroach",
    9: "Termite",
    10: "Centipede",
    11: "Fly",
    12: "Giraffe",
    13: "Tarantula",
    14: "Fire bug",
    15: "Tick",
    16: "Moth",
    17: "Millipede",
    18: "Mantis",
    19: "Dragonfly"
}

def load_image(image_file):
    return Image.open(image_file)

def detect_green_areas(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask

def calculate_area(mask, pixel_to_mm_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.contourArea(cnt) * (pixel_to_mm_ratio ** 2) for cnt in contours if cv2.contourArea(cnt) > 500]

def extract_number_from_filename(filename):
    match = re.search(r'_0*(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    st.title('Leaf Area Analysis Tool')
    
    uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files
