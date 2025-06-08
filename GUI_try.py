import streamlit as st
import numpy as np
#import pandas as pd
import cv2
from PIL import Image
#import tensorflow as tf
#import scipy as sp
#import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

# MAIN BACKGROUND 
st.markdown("""
<style>
    .stApp {
        background-color: #97D4F1; 
        min-height: 100vh;
    }
    
    /* Force sidebar styling dengan multiple selectors */
    section[data-testid="stSidebar"] {
        background-color: #C0DE7B !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background-color: #C0DE7B !important;
        color: white !important;
    }
    
    section[data-testid="stSidebar"] h2 {
        color: white !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        background-color: #79B425 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
    }

    /* ✅ HOVER dengan text forest green */
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: #FFFFFF !important;  /* Background putih */
        color: #228B22 !important;             /* Text forest green */
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Fallback untuk versi lama */
    .css-1d391kg {
        background-color: #C0DE7B !important;
        color: white !important;
    }
    
    .css-1d391kg .stButton > button {
        background-color: #79B425 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)


# ==== DEF EVERY PROCESS ====
# PRE-PROCESSING
# shape normalization
def crop_using_threshold(img, threshold=0.1):
    #Pastikan input dalam format RGB
    if img.ndim == 3:
       gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Normalisasi ke [0,1]
    norm_img = gray / 255.0

    # Threshold untuk membuat mask biner
    mask = norm_img > threshold

    # Cari bounding box dari mask
    coords = np.argwhere(mask)
    if coords.size == 0:
        return img  # fallback jika citra blank

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop citra asli (bukan grayscale!)
    cropped = img[y_min:y_max+1, x_min:x_max+1]
    return cropped

# resize
def prepare_image(img_path, target_size=(456, 456)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped = crop_using_threshold(img)
    resized = cv2.resize(cropped, target_size)
    return resized

# color normalization
def is_rgb_close(current_rgb, target_rgb, tolerance=15):
    return all(abs(c - t) < tolerance for c, t in zip(current_rgb, target_rgb))

def color_norm(image_np, target_rgb=(131.19, 65.04, 14.84), tolerance=15):
    pixels = image_np.astype(np.float32)
    mean_rgb = np.mean(pixels, axis=(0, 1))

    if is_rgb_close(mean_rgb, target_rgb, tolerance):
        print("⏭️ Skipped color normalization (already close to target).")
        return image_np

    Ri, Gi, Bi = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

    if mean_rgb[0] != 0:
        Ri = np.clip((Ri / mean_rgb[0]) * target_rgb[0], 0, 255)
    if mean_rgb[1] != 0:
        Gi = np.clip((Gi / mean_rgb[1]) * target_rgb[1], 0, 255)
    if mean_rgb[2] != 0:
        Bi = np.clip((Bi / mean_rgb[2]) * target_rgb[2], 0, 255)

    return np.dstack((Ri, Gi, Bi)).astype(np.uint8)

# CLAHE
def contrast_enhance(img_clahe):
  clahe = cv2.createCLAHE(clipLimit= 1.0, tileGridSize=(15, 15))

  Red = img_clahe[...,0]
  Green = img_clahe[...,1]
  Blue = img_clahe[...,2]

  Green_fix = clahe.apply(Green)
  new_img = np.stack([Red, Green_fix, Blue], axis=2)
  return new_img






## STREAMLIT VISUALIZATION
# Inisialisasi
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_page(page_name):
    st.session_state.page = page_name

# CSS
st.markdown("""
<style>
    .appview-container .main {
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    
    div.stButton > button {
        display: block;
        margin: 0 auto;
        width: auto;
        min-width: 200px;
    }
</style>
""", unsafe_allow_html=True)

# Routing

# ==== HOME PAGE ====
if st.session_state.page == "home":
    st.markdown("""
    <div style="background-color: #79B425; padding: 20px; border-radius: 10px; text-align: center;">
        <p style="font-size: 24px; margin-bottom: 2px; margin-top: 0; color: #FFFFFF;">Welcome to</p>
        <h1 style="font-size: 40px; margin-top: 0; color: #FFFFFF;">Diabetic Retinopathy Classification</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        detect_clicked = st.button("DETECT DR NOW!", key="detect")
        
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        learn_clicked = st.button("Learn More", key="learn")

    # Handle clicks SETELAH semua button didefinisikan
    if detect_clicked:
        st.session_state.page = "main"
        st.rerun()  # Paksa rerun
        
    if learn_clicked:
        st.session_state.page = "learn"
        st.rerun()  # Paksa rerun

# ==== MAIN PAGE ====
elif st.session_state.page == "main":
    st.title("INPUT IMAGE")

    # === SIDEBAR NAVIGASI ===
    with st.sidebar:
        st.header("📍 Navigation")
        
        if st.button("Home", key="nav_home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

        if st.button("Upload Image", key="nav_main_prep", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

        if st.button("Pre-Processing", key="nav_prepro_prep", use_container_width=True):
            st.session_state.page = "preprocessing"
            st.rerun()
            
        if st.button("Learn More", key="nav_learn", use_container_width=True):
            st.session_state.page = "learn"
            st.rerun()
    
    # === MAIN CONTENT ===
    st.subheader("📤 Upload Retinal Image")
    # File uploader
    uploaded_file = st.file_uploader(
    "Choose a retinal image file",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
    )

    # Handle uploaded file
    if uploaded_file is not None:
        # Display uploaded image
        st.success("✅ Image uploaded successfully!")
        # simpan uploaded image
        st.session_state.uploaded_file = uploaded_file
        st.session_state.uploaded_image = Image.open(uploaded_file)

        # Read image to get dimensions
        image = Image.open(uploaded_file)
        width, height = image.size
    
        # Create columns for better layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(
                uploaded_file, 
                caption=f"Uploaded: {uploaded_file.name}",
                use_column_width=True
            )
        with col2:
            st.markdown("**Image Info:**")
            st.write(f"📄 **Name:** {uploaded_file.name}")
            st.write(f"📊 **Size:** {width} x {height} pixels")  # Dimensi dalam piksel
            st.write(f"🎯 **Type:** {uploaded_file.type}")
    
            
    # Button to next step
    preprocessing_clicked = st.button("➡️ Pre-processing", key="preprocessing", type="primary", use_container_width=True)
    if preprocessing_clicked:
        st.session_state.page = "preprocessing"
        st.rerun()
    
    else:
        st.info("👆 Please upload a retinal image to get started")

# ==== PREPROCESSING PAGE ====
elif st.session_state.page == "preprocessing":
    st.title("Image Pre-Processing")

    st.subheader("Uploaded Image")
    # menampilkan uploaded file
    if 'uploaded_file' in st.session_state:
        uploaded_file = st.session_state.uploaded_file
        # size display image
        st.image(
            uploaded_file,
            caption = f"Uploaded Image",
            width = 400
        )
    else:
        st.warning("⚠️ No image found. Please upload an image first.")
        if st.button("← Go to Upload"):
            st.session_state.page = "main"
            st.rerun()
    
    # === SIDEBAR NAVIGASI ===
    with st.sidebar:
        st.header("📍 Navigation")
        
        if st.button("Home", key="nav_home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

        if st.button("Upload Image", key="nav_main_prep", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()

        if st.button("Pre-Processing", key="nav_prepro_prep", use_container_width=True):
            st.session_state.page = "preprocessing"
            st.rerun()
            
        if st.button("Learn More", key="nav_learn", use_container_width=True):
            st.session_state.page = "learn"
            st.rerun()
    
    # === CONTENT === 
    st.subheader("Shape Normalization")
    if 'uploaded_file' in st.session_state:
        # convert image to numpy array
        img_array = np.array(st.session_state.uploaded_image)
        # apply shape norm
        cropped_img = crop_using_threshold(img_array)

        # display result
        st.image(cropped_img, caption="Shape Normalized Image", width=400)
        st.write(f"Image Size : {cropped_img.shape[1]} x {cropped_img.shape[0]} pixels")
    

    st.subheader ("Resize")
    if 'uploaded_file' in st.session_state:
        img_array = np.array(st.session_state.uploaded_image)
        # apply shape norm
        cropped_img = crop_using_threshold(img_array)
        # apply resize 
        target_size = (456,456)
        resized_img = cv2.resize(cropped_img, target_size)

        # display result
        st.image(resized_img, caption="Resized Image", width=500)
        st.write(f"Image Size : {resized_img.shape[1]} x {resized_img.shape[0]} pixels")


    st.subheader ("Color Normalization")
    if 'uploaded_file' in st.session_state:
        img_array = np.array(st.session_state.uploaded_image)
        # apply shape norm
        cropped_img = crop_using_threshold(img_array)
        # apply resize 
        target_size = (456,456)
        resized_img = cv2.resize(cropped_img, target_size)
        # apply color norm
        color_normalized_img = color_norm(resized_img)

        # display result
        st.image(color_normalized_img, caption="Color Normalized Image", width=400)
        st.write(f"Image Size: {color_normalized_img.shape[1]} x {color_normalized_img.shape[0]} pixels")

    st.subheader ("CLAHE")
    if 'uploaded_file' in st.session_state:
        img_array = np.array(st.session_state.uploaded_image)
        # apply shape norm
        cropped_img = crop_using_threshold(img_array)
        # apply resize 
        target_size = (456,456)
        resized_img = cv2.resize(cropped_img, target_size)
        # apply color norm
        color_normalized_img = color_norm(resized_img)
        # apply clahe
        clahe_img = contrast_enhance(color_normalized_img)

        # display result
        st.image(clahe_img, caption="CLAHE Image", width=400)
        st.write(f"Image Size: {clahe_img.shape[1]} x {clahe_img.shape[0]} pixels")

    st.subheader ("final image") #side by side before vs after


# ==== LEARN PAGE ====
elif st.session_state.page == "learn":
    st.title("Learn about the app step-by-step")
    
    back_clicked = st.button("← Back to Home", key="back_learn")
    if back_clicked:
        st.session_state.page = "home"
        st.rerun()
       
