import os
import mlflow
from codecarbon import OfflineEmissionsTracker
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    auc,
    roc_curve,
    ConfusionMatrixDisplay,
)
import numpy as np

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

path_immagini_melanoma = "../../data/processed/train"
batch_size_value = 1024
batch_img_size = 224

tracker = OfflineEmissionsTracker(country_iso_code="CAN")
tracker.start()

data = tf.keras.utils.image_dataset_from_directory(
    path_immagini_melanoma,
    batch_size=batch_size_value,
    color_mode="rgb",
    image_size=(batch_img_size, batch_img_size),
    interpolation="bilinear",
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()

train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)

print("Data_size:", len(data))
print("Data_type:", type(data))
print("train_size:", train_size)
print("test_size:", test_size)
print("val_size:", val_size)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

mlflow.set_experiment("CNN 1024 batch")

with mlflow.start_run() as run:
    model = Sequential()
    model.add(
        Conv2D(
            16,
            (3, 3),
            1,
            activation="relu",
            input_shape=(batch_img_size, batch_img_size, 3),
        )
    )
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), 1, activation="relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), 1, activation="relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

    print(model.summary())

    logdir = "logs"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    hist = model.fit(
        train, epochs=20, validation_data=val, callbacks=[tensorboard_callback]
    )

    mlflow.tensorflow.log_model(model, "Convolutional Neural Network (CNN)")

    fig = plt.figure()
    plt.plot(hist.history["loss"], color="teal", label="loss")
    plt.plot(hist.history["val_loss"], color="orange", label="val_loss")
    fig.suptitle("Loss", fontsize=20)
    plt.legend(loc="upper left")
    mlflow.log_figure(fig, "Loss.png")

    fig = plt.figure()
    plt.plot(hist.history["accuracy"], color="teal", label="accuracy")
    plt.plot(hist.history["val_accuracy"], color="orange", label="val_accuracy")
    fig.suptitle("Accuracy", fontsize=20)
    plt.legend(loc="upper left")
    mlflow.log_figure(fig, "Accuracy.png")

    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    y_true = []
    y_pred = []

    model.save(os.path.join("../../models", "melanomaCNNClassifier.h5"))

    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
        y_true.extend(y)
        y_pred.extend(yhat)

    mlflow.log_metric("Precision", round(pre.result().numpy(), 2))
    mlflow.log_metric("Recall", round(re.result().numpy(), 2))
    mlflow.log_metric("Accuracy", round(acc.result().numpy(), 2))

    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    print(classification_report(y_true, y_pred_binary))

    # Confusion Matrix
    cm = confusion_matrix(y_true, np.round(y_pred))
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")

tracker.stop()
