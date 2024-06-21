import streamlit as st
import cv2
import numpy as np
from PIL import Image
import re
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import zipfile
from io import BytesIO

# Define the mapping from file numbers to Ecotron names
ecotron_names = {
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

# List of devices without plants
devices_without_plants = ["Cockroach", "Giraffe", "Fire bug", "Fly"]

# Function to load and conditionally crop an image
def load_image(image_file, crop_date_cutoff):
    image = Image.open(image_file)
    folder_date_str = os.path.basename(os.path.dirname(image_file))
    image_date = datetime.strptime(folder_date_str, '%B-%d-%Y-%I%p')
    if image_date >= crop_date_cutoff:
        if image.size == (4284, 5712):
            image = image.crop((0, 0, 4284, 4628))
    return image

# Function to detect green areas using HSV thresholding and morphological operations
def detect_green_areas(image, ecotron_name=None, timepoint=None):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply more stringent thresholding for Centipede and Tick on 2024-06-21 11AM and 2024-06-20 11AM
    if (ecotron_name == "Centipede" and timepoint == datetime(2024, 6, 21, 11)) or \
       (ecotron_name == "Tick" and timepoint in [datetime(2024, 6, 21, 11), datetime(2024, 6, 20, 11)]):
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([80, 255, 255])
    else:
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([85, 255, 255])
    
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    h, w = refined_mask.shape
    center = (w // 2, h // 2)
    radius = int(min(center) * 0.7)
    circular_mask = np.zeros_like(refined_mask)
    cv2.circle(circular_mask, center, radius, 255, -1)
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
    clean_mask = np.zeros_like(refined_mask)
    for cnt in large_contours:
        mask_temp = np.zeros_like(refined_mask)
        cv2.drawContours(mask_temp, [cnt], -1, 255, thickness=cv2.FILLED)
        if cv2.countNonZero(cv2.bitwise_and(mask_temp, circular_mask)) > 0:
            cv2.drawContours(clean_mask, [cnt], -1, 255, thickness=cv2.FILLED)
    return clean_mask

# Function to calculate the area of detected green regions
def calculate_area(mask, pixel_to_mm_ratio):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = sorted([cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 500], reverse=True)
    areas_mm2 = [area * (pixel_to_mm_ratio ** 2) for area in areas]
    areas_cm2 = [area_mm2 / 100 for area_mm2 in areas_mm2]
    if sum(areas_cm2) < 10:
        return [0, 0]
    return areas_cm2[:2]

# Function to extract the ecotron number from the filename
def extract_number_from_filename(filename):
    match = re.search(r'_0*(\d+)_', filename)
    if match:
        return int(match.group(1))
    return None

# Function to extract the timepoint from the folder name
def extract_timepoint_from_foldername(foldername):
    match = re.search(r'([A-Za-z]+-\d{1,2}-\d{4}-\d{1,2}[apm]+)', foldername)
    if match:
        date_str = match.group(1)
        timepoint = datetime.strptime(date_str, '%B-%d-%Y-%I%p')
        return timepoint
    return None

# Function to process images and extract relevant data
def process_images(folder_path, pixel_to_mm_ratio, crop_date_cutoff):
    results = []
    image_mask_pairs = {}
    for folder in os.listdir(folder_path):
        timepoint = extract_timepoint_from_foldername(folder)
        folder_full_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_full_path) and timepoint:
            for filename in os.listdir(folder_full_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_file_path = os.path.join(folder_full_path, filename)
                    image = load_image(image_file_path, crop_date_cutoff)
                    ecotron_name = ecotron_names.get(extract_number_from_filename(filename), "Unknown Ecotron")
                    if ecotron_name in devices_without_plants:
                        continue
                    mask = detect_green_areas(image, ecotron_name, timepoint)
                    areas = calculate_area(mask, pixel_to_mm_ratio)
                    if ecotron_name not in image_mask_pairs:
                        image_mask_pairs[ecotron_name] = []
                    image_mask_pairs[ecotron_name].append((image, Image.fromarray(mask), timepoint))
                    for i, area in enumerate(areas):
                        results.append({
                            "Timepoint": timepoint,
                            "Ecotron": ecotron_name,
                            "Sub-Plant": f"{ecotron_name} {i+1}",
                            "Total Area (cm²)": area
                        })

    # Sort images chronologically for each Ecotron name
    for ecotron_name in image_mask_pairs:
        image_mask_pairs[ecotron_name].sort(key=lambda x: x[2])

    return results, image_mask_pairs

# Function to process images from a zip file and extract relevant data
def process_zip_file(zip_file, pixel_to_mm_ratio, crop_date_cutoff):
    results = []
    image_mask_pairs = {}
    with zipfile.ZipFile(zip_file) as z:
        for file_info in z.infolist():
            if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                with z.open(file_info) as image_file:
                    image = load_image(image_file, crop_date_cutoff)
                    filename = os.path.basename(file_info.filename)
                    ecotron_name = ecotron_names.get(extract_number_from_filename(filename), "Unknown Ecotron")
                    if ecotron_name in devices_without_plants:
                        continue
                    folder_date_str = os.path.basename(os.path.dirname(file_info.filename))
                    timepoint = extract_timepoint_from_foldername(folder_date_str)
                    if not timepoint:
                        continue
                    mask = detect_green_areas(image, ecotron_name, timepoint)
                    areas = calculate_area(mask, pixel_to_mm_ratio)
                    if ecotron_name not in image_mask_pairs:
                        image_mask_pairs[ecotron_name] = []
                    image_mask_pairs[ecotron_name].append((image, Image.fromarray(mask), timepoint))
                    for i, area in enumerate(areas):
                        results.append({
                            "Timepoint": timepoint,
                            "Ecotron": ecotron_name,
                            "Sub-Plant": f"{ecotron_name} {i+1}",
                            "Total Area (cm²)": area
                        })

    # Sort images chronologically for each Ecotron name
    for ecotron_name in image_mask_pairs:
        image_mask_pairs[ecotron_name].sort(key=lambda x: x[2])

    return results, image_mask_pairs

# Function to plot the growth over time
def plot_growth_over_time(df):
# Function to plot the growth over time
def plot_growth_over_time(df):
    fig = px.line(df, x='Timepoint', y='Total Area (cm²)', color='Sub-Plant', markers=True, line_shape='linear')
    fig.update_layout(
        title='Growth and Area Increase Over Time for Each Plant',
        xaxis_title='Timepoint',
        yaxis_title='Total Area (cm²)',
        showlegend=False
    )
    return fig

# Function to plot the total growth per day
def plot_total_growth_per_day(df):
    df_sum = df.groupby('Timepoint')['Total Area (cm²)'].sum().reset_index()
    fig = px.bar(df_sum, x='Timepoint', y='Total Area (cm²)', title="Total Growth Per Day", labels={'Timepoint': 'Date', 'Total Area (cm²)': 'Total Area (cm²)'})
    fig.update_yaxes(range=[4000, df_sum['Total Area (cm²)'].max() * 1.1])
    return fig

# Function to plot the total size of plants in each device over time
def plot_total_growth_per_device(df):
    df_sum = df.groupby(['Timepoint', 'Ecotron'])['Total Area (cm²)'].sum().reset_index()
    fig = px.line(df_sum, x='Timepoint', y='Total Area (cm²)', color='Ecotron', markers=True, line_shape='linear')
    fig.update_layout(
        title='Total Size of Plants in Each Device Over Time',
        xaxis_title='Timepoint',
        yaxis_title='Total Area (cm²)'
    )
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title('Leaf Area Analysis Tool')

    uploaded_file = st.file_uploader("Upload a zip file with photos", type=['zip'])
    pixel_to_mm_ratio = st.number_input('Enter the pixel to mm ratio:', min_value=0.01, max_value=100.0, value=0.12, step=0.01)
    crop_date_cutoff = datetime(2024, 6, 21, 11)

    if uploaded_file and st.button('Analyze'):
        with st.spinner('Processing images...'):
            with BytesIO(uploaded_file.read()) as zip_file:
                results, image_mask_pairs = process_zip_file(zip_file, pixel_to_mm_ratio, crop_date_cutoff)
            st.success('Processing complete!')
            df = pd.DataFrame(results)

            # Plot growth analysis
            if not df.empty:
                total_growth_device_fig = plot_total_growth_per_device(df)
                st.plotly_chart(total_growth_device_fig)
                
                total_growth_fig = plot_total_growth_per_day(df)
                st.plotly_chart(total_growth_fig)
                
                growth_fig = plot_growth_over_time(df)
                st.plotly_chart(growth_fig)
            
            # Display results as a table
            st.subheader("Detailed Results Table")
            if not df.empty:
                df = df.sort_values(by=['Ecotron', 'Timepoint'])
                st.table(df)
                st.download_button("Download data as CSV", df.to_csv(index=False), "plant_areas.csv", "text/csv")

        # Create expanders for each timepoint and display image and segmentation mask
        for ecotron_name, images_masks in image_mask_pairs.items():
            for image, mask, timepoint in images_masks:
                with st.expander(f"{ecotron_name} - {timepoint.strftime('%Y-%m-%d %I%p')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption=f"Original Image ({ecotron_name})", use_column_width=True)
                    with col2:
                        st.image(mask, caption=f"Segmentation Mask ({ecotron_name})", use_column_width=True)

if __name__ == '__main__':
    main()
