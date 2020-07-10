import dataset
import evaluation
import train
from config import Config

def main():
    data = dataset.Dataset(Config.DATASET)
    features, labels, tag2idx, tag_values = data.load_features_and_labels()

    train.NER(features, labels, tag2idx, tag_values)

    evaluation.Predicter(tag2idx, tag_values)

    print(features[0])
    print(labels[0])



if __name__ == "__main__":
    main()