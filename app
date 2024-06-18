import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_image(image_file):
    return Image.open(image_file)

def detect_green_areas(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    
    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=3)
    
    return mask

def calculate_area(mask, pixel_to_mm_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) * (pixel_to_mm_ratio ** 2) for cnt in contours if cv2.contourArea(cnt) > 500]
    return areas

def main():
    st.title('Leaf Area Analysis Tool')
    
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        pixel_to_mm_ratio = st.number_input('Enter the pixel to mm ratio:', min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        
        if st.button('Analyze'):
            mask = detect_green_areas(image)
            areas = calculate_area(mask, pixel_to_mm_ratio)
            
            # Display results
            st.image(mask, caption='Green Mask', use_column_width=True)
            st.write(f"Total leaf areas in mm²: {areas}")

if __name__ == '__main__':
    main()
