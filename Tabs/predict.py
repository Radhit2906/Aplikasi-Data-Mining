import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from function import predict

mappingDesil ={
    "Kurang Sejahtera": 1,
    "Mendekati Sejahtera":2,
    "Sejahtera":3,
    "Sangat Sejahtera":4

}

mappingPekerjaan = {
    "Pekerja Lepas": 0,
    "Kosong/belum bekerja": 1,
    "Pegawai Swasta": 2,
    "Wiraswasta": 3,
    "Pedagang": 4,
    "Pensiunan": 5,
    "Petani": 6,
    "PNS/TNI/Polri": 7
}
mappingJenisKelamin ={  
    'Laki-Laki': 0,
    'Perempuan' : 1
}

mappingPrioritas={
    "Normal": 1,
    "NIK Duplikat": 0    
}

mappingKepalaKeluarga={
    "Kepala Keluarga": 2,
    "Istri":1,
    "Anak":0,
    "Lainnya":3
}
def app(df, x, y):
    #judul 
    st.title("Prediksi Data")
    
    #deskripsi aplikasi
    col1, col2 = st.columns(2)

    with col1:
        DesilKesejahteraan = st.selectbox('Desil', ('Kurang Sejahtera', 'Mendekati Sejahtera', 'Sejahtera', 'Sangat Sejahtera'))

    with col1:
        JenisKelamin = st.selectbox('Jenis Kelamin', ('Laki-Laki', 'Perempuan'))

    with col1:
        PrioritasVerval = st.selectbox('Prioritas', ('Normal', 'NIK Duplikat'))

    with col2:
        HubunganKepalaKeluarga = st.selectbox('Hubungan', ('Kepala Keluarga', 'Istri', 'Anak', 'Lainnya'))

    with col2:
        Pekerjaan = st.selectbox('Pekerjaan', ('Pekerja Lepas','Kosong/belum bekerja', 'Pegawai Swasta','Wiraswasta','Pedagang','Pensiunan','Petani','PNS/TNI/Polri'))

    with col2:
        Usia2023 = st.text_input('Usia')

    # Mengubah string ke float
    DesilKesejahteraan = float(mappingDesil[DesilKesejahteraan])
    Pekerjaan = float(mappingPekerjaan[Pekerjaan])
    JenisKelamin=float(mappingJenisKelamin[JenisKelamin])
    PrioritasVerval =float(mappingPrioritas[PrioritasVerval])
    HubunganKepalaKeluarga = float(mappingKepalaKeluarga[HubunganKepalaKeluarga])

    features = [DesilKesejahteraan, JenisKelamin, PrioritasVerval, 
                HubunganKepalaKeluarga, Pekerjaan, Usia2023]

    # Button
    if st.button("Prediction"):
        prediction, score = predict(x, y, features)
        score = score
        st.info("Prediksi Berhasil")

        if prediction == 1:
            st.warning("Penduduk Berhak Menerima Sembako")
        else:
            st.success("Penduduk Tidak Berhak Menerima Sembako")

        st.write("Tingkat Akurasi : ", (score * 100), "%")

#load dataset untuk diprediksi
#pastikan sudah diubah kedalam desimal dahulu seperti 0 1 2 3

    def load_data(file):
        data = pd.read_csv(file)
        return data

    # Fungsi untuk melatih model dengan algoritma Decision Tree
    def train_decision_tree(X_train, y_train):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        return clf

    # Fungsi untuk membuat prediksi menggunakan model yang sudah dilatih
    def make_prediction(model, data):
        predictions = model.predict(data)
        return predictions
    
    st.write("")
    st.write("")
    st.write("")
    

# Halaman utama Streamlit
    st.write("Upload File untuk mengecek akurasi data anda")
    st.write("catatan : harus diubah ke dalam float terlebih dahulu")

    # Widget untuk mengunggah file CSV
    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])
    

    if uploaded_file is not None:
        st.sidebar.subheader("Data yang Diunggah")
        data = load_data(uploaded_file)
        st.write(data)

        # Memilih kolom target untuk prediksi
        target_column = st.sidebar.selectbox("Pilih Kolom Label", data.columns)

        # Memilih fitur (kolom) untuk melatih model
        feature_columns = st.sidebar.multiselect("Pilih Kolom Fitur", data.columns)

        # Memisahkan data menjadi fitur (X) dan target (y)
        X = data[feature_columns]
        y = data[target_column]

        # Melihat akurasi model sebelum cross-validation
        st.write("### Akurasi Model Sebelum Cross-Validation")
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        predictions = clf.predict(X)
        accuracy_before_cv = accuracy_score(y, predictions)
        st.write(f"Akurasi Model: {accuracy_before_cv:.2%}")


        st.write("### Akurasi Model Setelah Cross-Validation")

         # Melakukan cross-validation untuk mendapatkan akurasi model pada data keseluruhan
        clf = DecisionTreeClassifier()
        cv_scores = cross_val_score(clf, X, y, cv=5)  # You can adjust the number of folds as needed
        st.write(f"Akurasi Model (Cross-Validation): {cv_scores.mean():.2%}")


        # Memisahkan data menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Melatih model
        model = train_decision_tree(X_train, y_train)

        # Membuat prediksi
        st.subheader("Prediksi")
        predictions = make_prediction(model, X_test)
        accuracy_after_test = accuracy_score(y_test, predictions)


        # Menampilkan hasil prediksi
        st.write("Hasil Prediksi:")
        # Menghitung dan menampilkan akurasi model
        st.write(f"Akurasi Model: {accuracy_after_test:.2%}")

# Example usage
# df, x, y are assumed to be defined elsewhere in your code
# app(df, x, y)

