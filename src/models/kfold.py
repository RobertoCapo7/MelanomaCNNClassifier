import os
import mlflow
from codecarbon import OfflineEmissionsTracker
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
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

data = data.map(lambda x, y: (x / 255, y))

# Convert dataset to numpy arrays
X = []
y = []
for batch in data:
    images, labels = batch
    X.append(images)
    y.append(labels)
X = np.concatenate(X)
y = np.concatenate(y)

# Define K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True)

mlflow.set_experiment("CNN K-FoldCV")

# Lists to store evaluation metrics across all folds
all_precisions = []
all_recalls = []
all_accuracies = []

# Iterate through folds
for fold, (train_indices, val_indices) in enumerate(kfold.split(X, y)):
    print(f"Fold {fold+1}")

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size_value)
    val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size_value)

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

        logdir = "logs"
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

        # Add model checkpoint callback to save the best model
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"best_model_fold_{fold+1}.keras",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )

        hist = model.fit(
            train, epochs=10, validation_data=val, callbacks=[tensorboard_callback, model_checkpoint_callback]
        )

        # Load the best model and evaluate on validation set
        best_model = tf.keras.models.load_model(f"best_model_fold_{fold+1}.keras")
        val_loss, val_accuracy = best_model.evaluate(val)
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

        # Log metrics
        mlflow.log_metric(f"Validation Loss Fold {fold+1}", val_loss)
        mlflow.log_metric(f"Validation Accuracy Fold {fold+1}", val_accuracy)

        # Evaluate on test set
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()
        y_true = []
        y_pred = []

        for batch in val:
            X_batch, y_batch = batch
            yhat = best_model.predict(X_batch)
            pre.update_state(y_batch, yhat)
            re.update_state(y_batch, yhat)
            acc.update_state(y_batch, yhat)
            y_true.extend(y_batch)
            y_pred.extend(yhat)

        precision = pre.result().numpy()
        recall = re.result().numpy()
        accuracy = acc.result().numpy()

        all_precisions.append(precision)
        all_recalls.append(recall)
        all_accuracies.append(accuracy)

        mlflow.log_metric(f"Precision_fold_{fold+1}", round(precision, 2))
        mlflow.log_metric(f"Recall_fold_{fold+1}", round(recall, 2))
        mlflow.log_metric(f"Accuracy_fold_{fold+1}", round(accuracy, 2))

        y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

        print(f"Precision_fold_{fold+1}: ", classification_report(y_true, y_pred_binary))

        # Confusion Matrix
        cm = confusion_matrix(y_true, np.round(y_pred))
        cm_display = ConfusionMatrixDisplay(cm).plot()
        plt.title('Confusion Matrix')
        plt.savefig(f'confusionMatrix_fold_{fold + 1}.png')
        mlflow.log_artifact(f'confusionMatrix_fold_{fold + 1}.png')

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(f'roc_curve_fold_{fold + 1}.png')
        mlflow.log_artifact(f'roc_curve_fold_{fold + 1}.png')

# Aggregate evaluation results
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_accuracy = np.mean(all_accuracies)

mlflow.log_metric("Average Precision", round(avg_precision, 2))
mlflow.log_metric("Average Recall", round(avg_recall, 2))
mlflow.log_metric("Average Accuracy", round(avg_accuracy, 2))

tracker.stop()
