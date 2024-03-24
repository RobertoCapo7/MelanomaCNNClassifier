import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

st.markdown("# MelanomaCNNClassifier🔬")
st.markdown(
    "_Questo caso d'uso presenta un classificatore avanzato basato su Convolutional Neural Network (CNN), "
    "progettato per distinguere in modo preciso tra nevi cutanei benigni e maligni. Il modello è stato "
    "addestrato su un ampio set di dati di immagini dermatoscopiche, permettendo al sistema di apprendere le caratteristiche "
    "distintive associate ai melanomi._"
)
batch_img_size = 224
uploaded_file = st.file_uploader("Scegli un'immagine...", type=["jpg", "png", "jpeg"])


def histogram_equalization(img_in):
    # segregate color streams
    b, g, r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    # calculate cdf
    cdf_b = np.cumsum(h_b)
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)

    # mask all pixels with value=0 and replace it with mean of the pixel values
    cdf_m_b = np.ma.masked_equal(cdf_b, 0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min()) * 255 / (cdf_m_b.max() - cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b, 0).astype("uint8")

    cdf_m_g = np.ma.masked_equal(cdf_g, 0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min()) * 255 / (cdf_m_g.max() - cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g, 0).astype("uint8")

    cdf_m_r = np.ma.masked_equal(cdf_r, 0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min()) * 255 / (cdf_m_r.max() - cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r, 0).astype("uint8")
    # merge the images in the three channels
    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]

    img_out = cv2.merge((img_b, img_g, img_r))
    # validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    cv2.merge((equ_b, equ_g, equ_r))

    return img_out


if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    img_np = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    st.image(pil_image, caption="Immagine caricata", use_column_width=True)
    resize = cv2.resize(img_bgr, (batch_img_size, batch_img_size))
    resize = histogram_equalization(resize)
    model = tf.keras.models.load_model("../../models/melanomaCNNClassifier.h5")
    pred = model.predict(np.expand_dims(resize / 255, 0))
    if pred[0] < 0.5:
        classificazione = ":green[**Benigno**]"
    else:
        classificazione = ":red[**Maligno**]"
    st.markdown(f"# Predizione: {classificazione}")
    pred = np.append(pred, 1 - pred[0])
    fig, ax = plt.subplots()
    sns.barplot(x=["Maligno", "Benigno"], y=pred, ax=ax)
    st.pyplot(fig)
