import streamlit as st
#import numpy as np
#import pandas as pd
#import cv2
#from PIL import Image
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
        /* Atau bisa pakai gradient */
        /* background: linear-gradient(135deg, #d4fc79, #96e6a1); */
        min-height: 100vh;
    }
</style>
""", unsafe_allow_html=True)


## DEF EVERY PROCESS

# PRE-PROCESSING
# shape normalization
#def crop_using_threshold(img, threshold=0.1):
    # Pastikan input dalam format RGB
#    if img.ndim == 3:
#       gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#    else:
#        gray = img.copy()
    
#    Normalisasi ke [0,1]
#    norm_img = gray / 255.0

    # Threshold untuk membuat mask biner
#    mask = norm_img > threshold

    # Cari bounding box dari mask
#    coords = np.argwhere(mask)
#    if coords.size == 0:
#        return img  # fallback jika citra blank

#    y_min, x_min = coords.min(axis=0)
#    y_max, x_max = coords.max(axis=0)

    # Crop citra asli (bukan grayscale!)
#    cropped = img[y_min:y_max+1, x_min:x_max+1]
#    return cropped

# resize
#def prepare_image(img_path, target_size=(456, 456)):
#    img = cv2.imread(img_path)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    cropped = crop_using_threshold(img)
#    resized = cv2.resize(cropped, target_size)
#    return resized

# color normalization
#def is_rgb_close(current_rgb, target_rgb, tolerance=15):
#    return all(abs(c - t) < tolerance for c, t in zip(current_rgb, target_rgb))

#def color_norm(image_np, target_rgb=(131.19, 65.04, 14.84), tolerance=15):
#    pixels = image_np.astype(np.float32)
#    mean_rgb = np.mean(pixels, axis=(0, 1))

#    if is_rgb_close(mean_rgb, target_rgb, tolerance):
#        print("⏭️ Skipped color normalization (already close to target).")
#        return image_np

#    Ri, Gi, Bi = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

#    if mean_rgb[0] != 0:
#        Ri = np.clip((Ri / mean_rgb[0]) * target_rgb[0], 0, 255)
#    if mean_rgb[1] != 0:
#        Gi = np.clip((Gi / mean_rgb[1]) * target_rgb[1], 0, 255)
#    if mean_rgb[2] != 0:
#        Bi = np.clip((Bi / mean_rgb[2]) * target_rgb[2], 0, 255)

#    return np.dstack((Ri, Gi, Bi)).astype(np.uint8)

# CLAHE
#def contrast_enhance(img_clahe):
#  clahe = cv2.createCLAHE(clipLimit= 1.0, tileGridSize=(15, 15))

#  Red = img_clahe[...,0]
#  Green = img_clahe[...,1]
#  Blue = img_clahe[...,2]

#  Green_fix = clahe.apply(Green)
#  new_img = np.stack([Red, Green_fix, Blue], axis=2)
#  return new_img






## STREAMLIT VISUALIZATION
# Inisialisasi halaman default
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_page(page_name):
    st.session_state.page = page_name

# box welcome to
st.markdown("""
<div style="background-color: #79B425; padding: 20px; border-radius: 10px; text-align: center;">
    <p style="font-size: 24px; margin-bottom: 2px; margin-top: 0; color: #FFFFFF; ">Welcome to</p>
    <h1 style="font-size: 40px; margin-top: 0; color: #FFFFFF; ">Diabetic Retinopathy Classification</h1>
</div>
""", unsafe_allow_html=True)

# Tombol navigasi sejajar
col1, col2, col3 = st.columns([1, 2, 1])  # Kolom tengah lebih lebar sebagai spacer
with col1:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    if st.button("DETECT DR NOW!"):
        st.session_state.page = "page1"
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    pass  # Kolom kosong sebagai spacer
with col3:
    st.markdown("<div style='text-align: center;'<", unsafe_allow_html=True)
    if st.button("learn more"):
        st.session_state.page = "page2"
    st.markdown("</div>", unsafe_allow_html=True)


# Konten halaman
if st.session_state.page == "home":
    pass
elif st.session_state.page == "page1":
    pass  # kosong, tidak menampilkan apapun
elif st.session_state.page == "page2":
    pass  # kosong, tidak menampilkan apapun

