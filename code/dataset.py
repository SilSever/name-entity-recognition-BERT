import pandas as pd


class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path

        self.n_sent = 1
        self.data = pd.read_csv(self.file_path, encoding="latin1").fillna(method="ffill")
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

        # self.features, self.labels = self.load_features_and_labels()

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    def _tag2idx(self):
        tag_values = list(set(self.data["Tag"].values))
        tag_values.append("PAD")
        tag2idx = {t: i for i, t in enumerate(tag_values)}
        return tag2idx, tag_values

    def load_features_and_labels(self):
        features = [[word[0] for word in sentence] for sentence in self.sentences]
        labels = [[s[2] for s in sentence] for sentence in self.sentences]
        tag2idx, tag_values = self._tag2idx()
        return features, labels, tag2idx, tag_values

