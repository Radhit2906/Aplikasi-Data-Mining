import streamlit as st
import pandas as pd
import numpy as np


import streamlit as st

def home_page():
    st.title("Home Page")
    st.write("Selamat datang di halaman utama!")
    st.title("""
# Klasifikasi Penduduk (Web Apps)
         Aplikasi berbasis web untuk memprediksi(klasifikasi) 
         Penduduk yang Berhak Menerima Sembako
         
"""

)

def program_page():
    st.title("Jalankan Program")
    st.sidebar.write('Upload File Anda (CSV)')
    upload_file = st.sidebar.file_uploader('Upload File CSV', type=["csv"])
    if upload_file is not None:
        inputan = pd.read_csv(upload_file)
    else:
        def input_user():
            desil = st.sidebar.selectbox('Desil', ('1','2','3','4'))
            prioritas = st.sidebar.selectbox('Prioritas',('Normal','NIK Duplikat'))
            jenis_kelamin = st.sidebar.selectbox('Jenis Kelamin', ('Laki-laki','Perempuan'))
            umur = st.sidebar.number_input('Umur')
            Hubungan_kepalakeluarga = st.sidebar.text_input('Hubungan dengan Kepala Keluarga')
            pekerjaan = st.sidebar.selectbox('pekerjaan', ('Pekerja lepas','Tidak/belum bekerja'))
            data = {'desil' : desil,
                'prioritas' : prioritas,
                'jenis_kelamin' : jenis_kelamin,
                'umur': umur,
                'Hubungan_kepalakeluarga' : Hubungan_kepalakeluarga,
                'pekerjaan' : pekerjaan}
            fitur = pd.DataFrame(data, index[0])
            return fitur
        inputan = input_user()

    #menggabungkan inputan dan dataset
    penduduk_raw = pd.read_csv('DatasetCoba1')
    penduduk = penduduk_raw.drop(columns=['No'])
    penduduk = penduduk_raw.drop(columns=['Kecamatan'])
    df = pd.concat([inputan, penduduk], axis=0)

    #Encode untuk numeric
    encode = ['Prioritas','JenisKelamin', 'HubunganKepalaKeluarga', 'Pekerjaan']
    for col in encode:
        dummy = pd.get_dummmies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    #ambil baris pertama 
    df = df[:1]

    #menampilkan inputan
    st.subheader('Parameter Inputan')

    if upload_file is not None:
        st.write(df)
    else:
        st.write('Menunggu  file csv untuk diupload. Ini masih menggunakan sample awal')
        st.write(df)


def contact_page():
    st.title("Contact Page")
    st.write("Hubungi kami di sini.")

# Fungsi untuk membuat tautan ke halaman terkait
def page_link(label, page_function):
    return f"[{label}](http://localhost:8501/{page_function.__name__})"

# Pilihan halaman
pages = {
    "Home": home_page,
    "Program": program_page,
    "About": contact_page,
}

# Sidebar untuk navigasi
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Select Page", list(pages.keys()))

# Tampilkan halaman yang dipilih
pages[selected_page]()
#--------------------------------------------------------

#Load Model