import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Menambahkan import library numpy
# Fungsi untuk menampilkan chart

def app():
    # Streamlit title and header
    
    st.title("Aplikasi Prediksi Klasifikasi Pembagian Sembako")

    st.write("Data yang digunakan")
    st.write("Data Sebelum di encoding")
    data = pd.read_csv('Datayangdipakai.csv')
    
    # Remove the following line, as it is not necessary and may cause an error
    # DataFrame.to_csv('Dataprojek', index=False)

    
    df = pd.DataFrame(data)
    st.dataframe(df)


    st.write("Atribut yang akan dipakai")
    data = data.drop(['No', 'Kecamatan'], axis=1)
    st.dataframe(data.head())

    features = data.columns[:-1]
    st.write(features)

    #Data Setelah di encoding

    show_chart(data) # Menggunakan df bukan data
    st.write("<p style='font-size: 20px;'>Berdasarkan chart di atas berisi target atau label yang digunakan pada aplikasi prediksi ini</p>",
             unsafe_allow_html=True)  # Menambahkan unsafe_allow_html untuk memungkinkan HTML
             
    st.subheader("Data setelah di encoding")
    data1 = pd.read_csv('Databarulagi1.csv')
    df1 = pd.DataFrame(data1)
    st.dataframe(df1)

def show_chart(data):

    st.title("Chart Atribut Penerima Sembako / Target")
    # Atribut yang akan divisualisasikan
    atribut = "PenerimaSEMBAKO"
    
    # Menggunakan Matplotlib untuk membuat chart
    fig, ax = plt.subplots()
    counts = data[atribut].value_counts()
    counts.plot(kind='bar', ax=ax, color='red')
    ax.set_ylabel("Jumlah Sample")
# Menambahkan jumlah pasti pada label y
    for i, v in enumerate(counts):
        ax.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    st.pyplot(fig)




