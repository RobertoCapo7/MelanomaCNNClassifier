import shutil
from src.data import downloadKaggleDataset
import warnings

# Ignora la specifica avvertenza che corrisponde al test "HTTPResponse.getheaders() is deprecated"
warnings.filterwarnings("ignore", message="HTTPResponse.getheaders() is deprecated")


def testDatasetPrecedentementeScaricato():
    assert (
        downloadKaggleDataset.download_kaggle_dataset(
            downloadKaggleDataset.dataset_name, downloadKaggleDataset.path_data_raw
        )
        == 1
    )


def testDownloadDataset():
    assert (
        downloadKaggleDataset.download_kaggle_dataset(
            downloadKaggleDataset.dataset_name, "data"
        )
        == 0
    )
    shutil.rmtree("data/train")
    shutil.rmtree("data/test")
