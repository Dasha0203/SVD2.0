import yfinance as yf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.write(""" 
# Прелесть SVD (сингулярного разложения матрицы) в изображениях

""")




from skimage import io

url = st.text_input('Введите URL Вашего изображения:')

if url:
    image = io.imread(url)
    

    # Обработка изображения
    image_gray = image[:, :, 0]
    u, sing_values, v = np.linalg.svd(image_gray)
    sigma = np.diag(sing_values)
    top_k = st.slider('Выберите количество сингулярных чисел', min_value=1, max_value=min(image_gray.shape), value=17)

    trunc_u = u[:, :top_k]
    trunc_sigma = sigma[:top_k, :top_k]
    trunc_v = v[:top_k, :]

    # Визуализация результатов
    fig, ax = plt.subplots(1, 2)  # Фиксированное количество подграфиков (1 строка, 2 столбца)
    ax[0].imshow(image, cmap='grey')        
    ax[0].set_title('Исходное')
    ax[1].imshow(trunc_u @ trunc_sigma @ trunc_v, cmap='gray')
    ax[1].set_title('Обрезанное')

    st.pyplot(fig)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
