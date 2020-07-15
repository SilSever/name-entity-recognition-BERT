from typing import Tuple, List, Dict, Union

import pandas as pd
from sklearn.model_selection import train_test_split

import utils


class Dataset:
    def __init__(self, file_path: str) -> None:
        """
        Dataset init function
        :param file_path: dataset path
        """
        self.test_set = False
        self.file_path = file_path

        self.data = pd.read_csv(self.file_path, encoding="latin1").fillna(
            method="ffill"
        )

        self.grouped = self.data.groupby("Sentence #").apply(self.agg)
        self.sentences = [s for s in self.grouped]

    @staticmethod
    def agg(sent: Union) -> List:
        """
        From a sentence retrieve the respectively words, POS's, tag's
        :param sent: sentence in the dataset
        :return: a list of words, POS's, tag's
        """
        return [
            (w, p, t)
            for w, p, t in zip(
                sent["Word"].values.tolist(),
                sent["POS"].values.tolist(),
                sent["Tag"].values.tolist(),
            )
        ]

    def _tag2idx(self) -> Tuple[Dict, List]:
        """
        Compute the label's dictionary
        :return: Dict tag to id, label's dictionary
        """
        tag = list(set(self.data["Tag"].values))
        tag.append("PAD")
        tag2idx = {t: i for i, t in enumerate(tag)}
        return tag2idx, tag

    def load_train_test(self) -> Tuple[List, List, List, List, Dict, List]:
        """
        Compute the features and labels and puts them in a list
        in such a way to be tokenized in training step
        :return: features, labels, _tag2idx function
        """
        tag2idx, tag = self._tag2idx()
        test = []
        test_lab = []

        if self.test_set:
            features = [[field[0] for field in fields] for fields in self.sentences]
            labels = [[field[2] for field in fields] for fields in self.sentences]

        else:
            # features = [" ".join([field[0] for field in fields]) for fields in self.sentences]
            # labels = [" ".join([field[2] for field in fields]) for fields in self.sentences]

            features = [[field[0] for field in fields] for fields in self.sentences]
            labels = [[field[2] for field in fields] for fields in self.sentences]

            train, test, tr_labels, test_lab = train_test_split(
                features, labels, test_size=0.1, shuffle=False
            )

            utils.check_integrity(train, tr_labels, desc='Train')
            utils.check_integrity(test, test_lab, desc='Test')

            # features = [feature.split() for feature in train]
            # labels = [label.split() for label in tr_labels]

        return features, labels, test, test_lab, tag2idx, tag
