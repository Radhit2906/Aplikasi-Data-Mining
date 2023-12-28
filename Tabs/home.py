import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Menambahkan import library numpy
# Fungsi untuk menampilkan chart

def app():
    # Streamlit title and header
    
    st.title("Aplikasi Prediksi Klasifikasi Pembagian Sembako")

    #Memberi Jarak
    st.markdown('<br style="line-height:50px;">', unsafe_allow_html=True)

    st.subheader("Dataset")
    st.write("Dataset bersumber dari Kementerian Koordinator Pembangunan Manusia dan Kebudayaan. 2023.")
    st.write("Metode yang digunakan yaitu : Decision Tree")

    st.subheader("Tujuan")
    st.write("Untuk melakukan prediksi dalam klasifikasi penduduk yang layak mendapatkan bantuan sembako, sehingga bantuan sembako dapat sampai kepada orang yang tepat")
    
    
    #Proses
    st.subheader("Proses")
    st.subheader("Data Sebelum di encoding")
    data = pd.read_csv('Datayangdipakai.csv')
    
    # Remove the following line, as it is not necessary and may cause an error
    # DataFrame.to_csv('Dataprojek', index=False)

    
    df = pd.DataFrame(data)
    st.dataframe(df)

    #Melakukan drop pada No dan Kecamatan karena tidak dipakai

    st.subheader("Memilih Atribut yang dipakai")
    st.write("Melakukan drop kepada atribut yang tidak dipakai")
    st.code("data = data.drop(['No', 'Kecamatan'], axis=1)")
    data = data.drop(['No', 'Kecamatan'], axis=1)

    #Menampilkan data setelah dihapus
    st.write(data.head())

    st.subheader("Atribut yang akan dipakai")

    features = data.columns[:-1]
    st.write(features)

    #Mengelompokkan
    st.subheader("Mengelompokkan data")
    st.write("Data dikelompokkan menjadi 2 yaitu Numeric dan Kategori")
    kelompok =""" 
    
    numerical = []
    catgcols =[]

    #iterasi
    for col in df.columns:
        #jika tipe data int64
        if df[col].dtype=="int64":
            #maka numeric
            numerical.append(col)
        else:
            #selain itu
            catgcols.append(col)

    #iterasi
    for col in df.columns:
        #jika numeric
        if col in numerical:
            #jika ada yang missing atau kosong diisi dengan median/rata-rata dari kolom tsb
            df[col].fillna(df[col].median(),inplace=True)
        else:
            #jika bukan diisi dengan nilai modus
            df[col].fillna(df[col].mode()[0],inplace=True)

    """
    st.code(kelompok,language="python")

    #Data Setelah di encoding

    show_chart(data) # Menggunakan df bukan data
    st.write("<p style='font-size: 20px;'>Berdasarkan chart di atas berisi target atau label yang digunakan pada aplikasi prediksi ini</p>",
             unsafe_allow_html=True)  # Menambahkan unsafe_allow_html untuk memungkinkan HTML
             
    st.subheader("Data setelah di encoding")

    st.write("Proses dalam encoding") 
    code_to_display = """
        le = LabelEncoder()

        for col in catgcols:
            df[col] = le.fit_transform(df[col])
        """

    st.code(code_to_display, language="python")

    data1 = pd.read_csv('Databarulagi1.csv')
    df1 = pd.DataFrame(data1)
    st.dataframe(df1)

    st.subheader("Membagi Data")
    st.write("Membagi data untuk training dan testing")
    st.code("x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)")

    st.subheader("Membuat Decision Tree Classifier")

    dtc_display = """ 
    dtc = DecisionTreeClassifier(
 
    ccp_alpha=0.0, class_weight=None, criterion='entropy',
    max_depth=4, max_features=None, max_leaf_nodes=None,
    min_impurity_decrease=0.0, min_samples_leaf=1,
    min_samples_split=2, min_weight_fraction_leaf=0.0,
    random_state=42, splitter='best'
    )

    """
    st.code(dtc_display, language="python")

    st.write("Melatih model dengan x_train sebagai data training dan y_train sebagai label training")
    st.code("model = dtc.fit(x_train, y_train)")

    st.write("Cross Validation dengan cv atau lipatan bebas ")
    st.write("Cross Validation akan melakukan pelatihan dan pengujian model pada setiap iterasi cross-validation, dan mengembalikan nilai akurasi (atau skor yang dipilih) pada setiap iterasi. ")

    crs_display =""" 
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, x, y, cv=5)
    print("Cross-validated scores:", scores)
    print("Average accuracy:", scores.mean())

    """

    st.code(crs_display, language="python")

    st.write("Menghitung akurasi pada data pengujian ")
    st.code("dtc_acc = accuracy_score(y_test, dtc.predict(x_test,))")
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




