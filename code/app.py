import dataset
import evaluation
import train
from config import Config


def main() -> None:
    """
    Main function
    :return: None
    """
    data = dataset.Dataset(Config.DATASET)
    features, labels, test, test_lab, tag2idx, tag_values = data.load_train_test()

    ner = train.NER(features, labels, tag2idx, tag_values)
    model = ner.train()

    evaluation.Predicter(test, test_lab, tag2idx, tag_values, model)


if __name__ == "__main__":
    main()
