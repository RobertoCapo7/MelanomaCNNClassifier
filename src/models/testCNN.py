import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

path_modello = "../../models/melanomaCNNClassifier312.h5"
path_test_benign = "../../data/processed/test/Benign"
path_test_malignant = "../../data/processed/test/Malignant"
batch_img_size = 224

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


model = load_model(path_modello)

contError = 0
contImage = 0

for img_name in os.listdir(path_test_benign):
    img_path = os.path.join(path_test_benign, img_name)
    img = cv2.imread(img_path)
    resize = cv2.resize(img, (batch_img_size, batch_img_size))
    resize = histogram_equalization(resize)
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    contImage += 1
    if yhat > 0.5:
        contError += 1

for img_name in os.listdir(path_test_malignant):
    img_path = os.path.join(path_test_malignant, img_name)
    img = cv2.imread(img_path)
    resize = cv2.resize(img, (batch_img_size, batch_img_size))
    resize = histogram_equalization(resize)
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    contImage += 1
    if yhat < 0.5:
        contError += 1

print("Totale Immagini:", contImage)
print("Predizioni corrette", contImage - contError)
print("Predizioni errate:", contError)
