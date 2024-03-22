import os
import kaggle

dataset_name = "bhaveshmittal/melanoma-cancer-dataset"
path_data_raw = "../../data/raw"


def download_kaggle_dataset(dataset_name, path_data_raw):
    kaggle.api.authenticate()

    train_path = os.path.join(path_data_raw, "train")
    test_path = os.path.join(path_data_raw, "test")

    if os.path.exists(train_path) or os.path.exists(test_path):
        print("Il dataset train e test è già stato scaricato.")
        return 1
    else:
        try:
            kaggle.api.dataset_download_files(
                dataset_name, path=path_data_raw, unzip=True, force=True
            )
            print("Dataset scaricato ed estratto correttamente.")
        except Exception as e:
            print(f"Si è verificato un errore durante il download del dataset: {e}")
            return 1

    return 0


if __name__ == "__main__":
    download_kaggle_dataset(dataset_name, path_data_raw)
