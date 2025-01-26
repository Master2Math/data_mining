import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import io

# Load dataset
st.title('Klasifikasi Data dengan Random Forest')

with st.expander("Dataset"):
    data = pd.read_csv("Classification.csv")
    st.write(data)
    
    st.success('Informasi Dataset')
    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    
    st.success('Analisa Univariat')
    deskriptif = data.describe()

    st.write(deskriptif)

# Pastikan nama kolom tidak memiliki spasi tersembunyi
data.columns = data.columns.str.strip()
# 
# Pastikan nama kolom benar
target_col = 'Drug'
fitur_angka = ['Age', 'Na_to_K']

if target_col not in data.columns:
    st.error(f"Kolom '{target_col}' tidak ditemukan dalam dataset! Pastikan nama kolom sesuai.")
else:
    with st.expander('Visualisasi'):
        st.info('Visualisasi Per Column')

        # Histogram Age
        fig, ax = plt.subplots()
        sns.histplot(data['Age'], color='blue', kde=True)
        plt.xlabel('Age')
        st.pyplot(fig)

        # Histogram Na_to_K
        fig, ax = plt.subplots()
        sns.histplot(data['Na_to_K'], color='red', kde=True)
        plt.xlabel('Na_to_K')
        st.pyplot(fig)

        st.info('Korelasi Heatmap')
        matriks_korelasi = data[fitur_angka].corr()

        fig, ax = plt.subplots()
        sns.heatmap(matriks_korelasi, annot=True, cmap='RdBu', fmt='.2f')
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('Korelasi Antar Fitur Angka', fontsize=10)
        st.pyplot(fig)

        def plot_outlier(data, column):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.boxplot(y=data[column], ax=axes[0])
            axes[0].set_title(f'{column} - Box Plot')

            sns.histplot(data[column], kde=True, ax=axes[1])
            axes[1].set_title(f'{column} - Histogram')
            st.pyplot(fig)

        # Deteksi outlier untuk Age dan Na_to_K
        plot_outlier(data, 'Age')
        plot_outlier(data, 'Na_to_K')

        def remove_outlier(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            return data[(data[column] >= lower) & (data[column] <= upper)]

        st.success('Data Setelah Outlier')
        data = remove_outlier(data, 'Age')
        data = remove_outlier(data, 'Na_to_K')

    # Encoding categorical variables
    X = data[['Age', 'Na_to_K']]
    y = data['Drug']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Random Forest Classifier
    st.success('Apply Random Forest Classifier')
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    # Evaluasi Model
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy: {accuracy:.2f}')
    st.text('Classification Report:')
    st.text(classification_report(y_test, y_pred))

     # Prediction
    st.header('Prediksi Data Baru')

    # Input fields for prediction
    age_input = st.number_input('Masukkan Umur (Age)', min_value=0, max_value=100, value=25)
    na_to_k_input = st.number_input('Masukkan Rasio Na to K', min_value=0.0, max_value=30.0, value=15.0)

    # Predict button
    if st.button('Prediksi'):
        input_data = pd.DataFrame([[age_input, na_to_k_input]], columns=['Age', 'Na_to_K'])
        prediction = classifier.predict(input_data)

        # Display result
        st.write(f'Prediksi: {prediction[0]}')
