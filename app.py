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
    areas = [cv2.contourArea(cnt) * (pixel_to_mm_ratio ** 2) for cnt in contours if cv2.contourArea(cnt) * (pixel_to_mm_ratio ** 2) > 1e7]  # Only consider areas > 1e7 mm²
    return areas

def plot_areas(areas, insect_name):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(areas)), areas, color='green')
    plt.xlabel('Sub-Plant Number')
    plt.ylabel('Area (mm²)')
    plt.title(f'Area of Each Sub-Plant for {insect_name}')
    plt.grid(True)
    plt.tight_layout()
    return plt

def extract_number_from_filename(filename):
    match = re.search(r'_0*(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    st.title('Leaf Area Analysis Tool')
    
    uploaded_files = st.file_uploader("Choose images...", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    pixel_to_mm_ratio = st.number_input('Enter the pixel to mm ratio:', min_value=0.01, max_value=100.0, value=8.43, step=0.01)
    
    all_plants = {}  # Dictionary to store all plant areas by insect name
    
    if uploaded_files and st.button('Analyze'):
        for uploaded_file in uploaded_files:
            image_number = extract_number_from_filename(uploaded_file.name)
            insect_name = insect_names.get(image_number, "Unknown Insect")
            
            image = load_image(uploaded_file)
            st.image(image, caption=f'Uploaded Image: {insect_name}', use_column_width=True)
            
            mask = detect_green_areas(image)
            areas = calculate_area(mask, pixel_to_mm_ratio)
            
            if insect_name not in all_plants:
                all_plants[insect_name] = []
            all_plants[insect_name].extend(areas)
            
            if areas:
                plot = plot_areas(areas, insect_name)
                st.pyplot(plot)
        
        # Plotting comparative graph for all plants
        plt.figure(figsize=(15, 7))
        for idx, (name, areas) in enumerate(all_plants.items()):
            plt.bar(idx, sum(areas), label=name, color=plt.cm.tab20(idx / len(all_plants)))
        
        plt.xlabel('Insect Name')
        plt.ylabel('Total Area (mm²)')
        plt.title('Comparative Total Area of Plants Across All Images')
        plt.legend()
        plt.tight_layout()
        st.pyplot()

if __name__ == '__main__':
    main()
