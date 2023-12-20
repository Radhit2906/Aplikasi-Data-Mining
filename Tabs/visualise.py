import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn import tree
import streamlit as st

from function import train_model

def app(df, x, y):
    if df is None or df.empty:
        st.warning("Please upload a CSV file to get started.")
        return
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Halaman Visualisasi")

    if st.checkbox("Plot Confusion Matrix"):
        model, score = train_model(x, y)

        # Buat prediksi menggunakan model
        y_pred = model.predict(x)

        # Dapatkan matriks kebingungan
        cm = confusion_matrix(y, y_pred)

        # Tampilkan matriks kebingungan menggunakan ConfusionMatrixDisplay
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        cmd.plot(cmap='viridis', values_format='d', ax=plt.gca())
        
        st.pyplot()

    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x,y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=4, out_file=None, filled=True,rounded=True,
            feature_names=x.columns, class_names=['Tidak','Ya']
        )

        st.graphviz_chart(dot_data)
