import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re
import pandas as pd
import zipfile
from io import BytesIO

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

def calculate_area_and_count_leaves(mask, pixel_to_mm_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) * (pixel_to_mm_ratio ** 2) for cnt in contours if cv2.contourArea(cnt) * (pixel_to_mm_ratio ** 2) > 1e7]
    num_leaves = len([cnt for cnt in contours if cv2.contourArea(cnt) * (pixel_to_mm_ratio ** 2) > 1e7])
    return areas, num_leaves

def plot_areas(areas, insect_name):
    if not areas:
        return None  # Handle the case where no significant areas are found
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(areas)), areas, color='green')
    ax.set_xlabel('Sub-Plant Number')
    ax.set_ylabel('Area (mm²)')
    ax.set_title(f'Area of Each Sub-Plant for {insect_name}')
    ax.grid(True)
    plt.tight_layout()
    return fig

def extract_number_from_filename(filename):
    match = re.search(r'_0*(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

def process_images(image_files, pixel_to_mm_ratio):
    all_plants = {}  # Dictionary to store all plant areas by insect name
    results = []  # List to store results for the table

    for image_file in image_files:
        filename = image_file.name
        image_number = extract_number_from_filename(filename)
        insect_name = insect_names.get(image_number, "Unknown Insect")
        
        image = load_image(image_file)
        st.image(image, caption=f'Uploaded Image: {insect_name}', use_column_width=True)
        
        mask = detect_green_areas(image)
        areas, num_leaves = calculate_area_and_count_leaves(mask, pixel_to_mm_ratio)
        
        if insect_name not in all_plants:
            all_plants[insect_name] = []
        all_plants[insect_name].extend(areas)
        
        for i, area in enumerate(areas):
            sub_plant_label = f"{image_number}.{i+1}"
            results.append({"Insect Name": insect_name, f"Sub-Plant {sub_plant_label} Area (mm²)": area, f"Sub-Plant {sub_plant_label} Leaf Count": num_leaves})
        
        plot = plot_areas(areas, insect_name)
        if plot:
            st.pyplot(plot)
        else:
            st.write("No significant areas detected for", insect_name)
    
    return results, all_plants

def main():
    st.title('Leaf Area Analysis Tool')
    
    uploaded_files = st.file_uploader("Choose images or a zip file...", type=['jpg', 'jpeg', 'png', 'zip'], accept_multiple_files=True)
    pixel_to_mm_ratio = st.number_input('Enter the pixel to mm ratio:', min_value=0.01, max_value=100.0, value=8.43, step=0.01)
    
    if uploaded_files and st.button('Analyze'):
        image_files = []

        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith('.zip'):
                with zipfile.ZipFile(uploaded_file) as z:
                    for filename in z.namelist():
                        if filename.endswith(('.jpg', '.jpeg', '.png')):
                            with z.open(filename) as image_file:
                                image_files.append(BytesIO(image_file.read()))
            else:
                image_files.append(uploaded_file)
        
        results, all_plants = process_images(image_files, pixel_to_mm_ratio)
        
        # Display results as a table
        df = pd.DataFrame(results)
        st.table(df)
        
        # Preparing data for the stacked bar plot
        df_stacked = pd.DataFrame.from_dict(all_plants, orient='index').fillna(0).T
        fig, ax = plt.subplots(figsize=(15, 7))
        
        # Plotting stacked bars
        df_stacked.plot(kind='bar', stacked=True, ax=ax, colormap='tab20')
        
        ax.set_xlabel('Plant Position')
        ax.set_ylabel('Total Area (mm²)')
        ax.set_title('Comparative Total Area of Plants Across All Images')
        ax.legend(title='Insect Name')
        
        # Custom x-axis labels
        ax.set_xticklabels(['Left Plant', 'Right Plant'], rotation=0)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Download button for CSV
        st.download_button(
            label="Download data as CSV",
            data=df.to_csv(index=False),
            file_name='plant_areas.csv',
            mime='text/csv',
        )

if __name__ == '__main__':
    main()
