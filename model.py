import pandas as pd # membaca data 
import streamlit as st # untuk membuka file di streamlit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
import io 
from sklearn.model_selection import train_test_split


st.title('Prediksi Kadar EMISI CO2 Pada Kendaraan')
with st.expander("DataSet"):
    data = pd.read_csv("FuelConsumptionCo2.csv")
    st.write(data)

    st.success('Informasi Dataset')
    data1 = pd.DataFrame(data)
    buffer = io.StringIO()
    data1.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.success('Analisa Univariat')
    # count = jumlah data
    # mean (u) = nilai rata-rata
    # std (standar deviasi)  = simpangan (untuk membaca grafik) -> jika 0 pasti homogen atau seragam atau sama semuanya -> 68% berada pada 1 sd -> u +- s
            # 3.34 +- 1.4
            # 3.34 - 1.4 = 1.94
            # 3.34 + 1.4 = 4.74
            # 1 sd
    # min  = nilai terendah 
    # 25% (Q1) = kuarter 1
    # 50% (Q2) = kuarter 2
    # 75% (Q3) = kuarter 3
    # max (Q4) = kuarter 

    deskriptif = data.describe()
    st.write(deskriptif)

with st.expander('Visualisasi'):
    st.info('Visualisasi Per Column')

    fig,ax = plt.subplots()
    sns.histplot(data['ENGINESIZE'], color='blue')
    plt.xlabel('Engine Size')
    st.pyplot(fig)

    fig,ax = plt.subplots()
    sns.histplot(data['CYLINDERS'], color='red')
    plt.xlabel('Cylinders')
    st.pyplot(fig)

    st.info('Korelasi Heatmap')
    fitur_angka = ['MODELYEAR',
                   'ENGINESIZE',
                   'CYLINDERS',
                   'FUELCONSUMPTION_CITY',
                   'FUELCONSUMPTION_HWY',
                   'FUELCONSUMPTION_COMB',
                   'FUELCONSUMPTION_COMB_MPG',
                   'CO2EMISSIONS']
    matriks_korelasi = data[fitur_angka].corr()

    fig, ax = plt.subplots()
    sns.heatmap(matriks_korelasi, annot=True, cmap='RdBu')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.title('Korelasi Antar Fitur Angka', fontsize=8)
    st.pyplot(fig)

    def plot_outlier(data,column):
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        sns.boxplot(data[column])
        plt.title(f'{column} - Box Plot')

        plt.subplot(1,2,2)
        sns.histplot(data[column],kde=True)
        plt.title(f'{column} - Histogram')
    # st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(plot_outlier(data,'MODELYEAR'))
    st.pyplot(plot_outlier(data,'CYLINDERS'))
    st.pyplot(plot_outlier(data,'ENGINESIZE'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_CITY'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_HWY'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_COMB'))
    st.pyplot(plot_outlier(data,'FUELCONSUMPTION_COMB_MPG'))

    def remove_outlier(data,column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)

        IQR = Q3-Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        
        data = data[(data[column] >= lower) & (data[column] <= upper)]

        return data
    
    st.success('Data Setelah Outlier')
    data = remove_outlier(data, 'MODELYEAR')
    data = remove_outlier(data, 'CYLINDERS')
    data = remove_outlier(data, 'ENGINESIZE')
    data = remove_outlier(data, 'FUELCONSUMPTION_CITY')
    data = remove_outlier(data, 'FUELCONSUMPTION_HWY')
    data = remove_outlier(data, 'FUELCONSUMPTION_COMB')
    data = remove_outlier(data, 'FUELCONSUMPTION_COMB_MPG')

# SKIP
# SKIP Ketinggalan
# SKIP
    st.success('Apply Random Forest Regressor')
    rf_regressor = RandomForestRegressor(max_depth=2, random_state=1)
    rf_regressor.fit(X_train, y_train)

    # prediksi
    y_pred_rf = rf_regressor.predict(X_test)
    score = mean_absolute_error(y_test, y_pred_rf)
    st.write(score)



        


