import pathlib
import sys


class Config:

    BERT_MODEL = "bert-base-cased"

    running_on_colab = "google.colab" in sys.modules

    if running_on_colab:
        RESOURCES = pathlib.Path("/content/drive/My Drive/NER/resources")
        DATA = pathlib.Path("/content/drive/My Drive/NER/data")
    else:
        RESOURCES = pathlib.Path(__file__).resolve().parent.parent / "resources"
        DATA = pathlib.Path(__file__).resolve().parent.parent / "data"

    MODEL = RESOURCES / "model"
    DATASET = DATA / "1014_4361_bundle_archive" / "ner_dataset.csv"

    PREDICTION = RESOURCES / "prediction.txt"
