import streamlit as st
import cv2
import Classification
import pandas as pd

if __name__ == '__main__':    
    st.title("OPTIMASI SUPPORT VECTOR MACHINE MENGGUNAKAN ALGORITMA GENETIKA DAN GRID SEARCH UNTUK KLASIFIKASI KANKER PARU-PARU PADA CITRA COMPUTED TOMOGRAPHY SCAN")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])
    if uploaded_file is not None:
        image = cv2.imread("E://Coding//Skripsi//Dataset//brain_tumor_dataset//test//"+uploaded_file.name, cv2.IMREAD_GRAYSCALE)
        st.image(image, caption='Uploaded Gambar.', width=250)
        st.write("")
        st.write("Classifying...")
        
        klasifikasi = Classification.Classification()
        feature = klasifikasi.self_glcm(image)
        dffeature = pd.DataFrame(feature).T
        dffeature.columns=['contrast', 'energy', 'correlation', 'dissimilarity', 'homogeneity', 'ASM']
        df = pd.read_csv("E://Coding//Skripsi//bestfeature.csv", index_col=0)
        bestfeature = df['feature'].to_numpy()
        label = klasifikasi.label(dffeature, bestfeature)
        
        if label == 0:
            st.header("Prediksi gambar ini adalah Normal")
        if label == 1:
            st.header("Prediksi gambar ini adalah Abnormal")