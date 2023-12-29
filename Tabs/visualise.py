import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
from sklearn import tree
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from function import train_model

def app(df, x, y):
    if df is None or df.empty:
        st.warning("Please upload a CSV file to get started.")
        return
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.title("Visualization Page")

    if st.checkbox("Plot Confusion Matrix"):



        model, score = train_model(x, y)

        # Split the data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        # Train the model on the training set
        model.fit(x_train, y_train)

        # Make predictions using the model
        y_pred = model.predict(x_test)

        # Get the confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Calculate evaluation metrics
        true_positive = cm[1, 1]
        true_negative = cm[0, 0]
        false_positive = cm[0, 1]
        false_negative = cm[1, 0]

        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)

        # Display confusion matrix using ConfusionMatrixDisplay
        cmd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        cmd.plot(cmap='viridis', values_format='d', ax=plt.gca())
        
        # Create a DataFrame to display metrics
        data = {
            'Metric': ['Positives', 'Negatives', 'True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Precision', 'Recall', 'Accuracy'],
            'Value': [true_positive + false_positive, true_negative + false_negative, true_positive, true_negative, false_positive, false_negative, precision, recall, accuracy]
        }

        st.pyplot()

        df_metrics = pd.DataFrame(data)
        st.table(df_metrics)




        st.write("<p style='font-size: 20px;'><strong>Keterangan<strong></p>",
             unsafe_allow_html=True)
        
        st.write("**Positives**          : Jumlah total kasus positif dalam dataset dalam prediksi model")
        st.write("**Negatives**          : Jumlah total kasus negatif dalam dataset dalam prediksi model")
        st.write("**True Positives**     : Jumlah kasus di mana model benar-benar memprediksi kelas positif dan prediksinya benar.")
        st.write("**True Negatives**     : Jumlah kasus di mana model benar-benar memprediksi kelas negatif dan prediksinya benar.")
        st.write("**False Positives**    : Jumlah kasus di mana model memprediksi kelas positif, tetapi sebenarnya kelasnya negatif.")
        st.write("**False Negatives**    : Jumlah kasus di mana model memprediksi kelas negatif, tetapi sebenarnya kelasnya positif.")
        st.write("**Precision**          : mengukur sejauh mana prediksi positif model akurat.")
        st.write("**Recall**             : Recall mengukur sejauh mana model dapat menangkap atau mengidentifikasi semua kasus positif yang sebenarnya.")
        st.write("**Akurasi**            : mengukur sejauh mana model secara keseluruhan memprediksi dengan benar.")
        

        st.subheader("Classification Report")

        cr = classification_report(y_test, y_pred)
        st.text(cr)
        
    if st.checkbox("Plot Decision Tree"):
        model, score = train_model(x, y)
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=4, out_file=None, filled=True, rounded=True,
            feature_names=x.columns, class_names=['No', 'Yes']
        )

        st.graphviz_chart(dot_data)

# Example usage
# df, x, y are assumed to be defined and initialized before calling the app function
# app(df, x, y)
