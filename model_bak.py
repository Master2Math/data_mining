import streamlit as st
import pandas as pd # membaca data 
import matplotlib.pyplot as plt
import seaborn as sns
import io 

st.title('Apikasi Pengukur Kadar Emisi CO2')
with st.expander("DataSet"):
    data = pd.read_csv("FuelConsumptionCo2.csv")
    data

    st.write('Informasi Fitur')
    informasi = pd.DataFrame(data)
    informasi
    buffer = io.StringIO()
    informasi.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    st.success('Analisa Univariat')
    Statistika = data.describe()
    st.write(Statistika)
