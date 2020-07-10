import dataset
from config import Config

def main():
    data = dataset.Dataset(Config.DATASET)
    features, labels, tag2idx = data.tokens_and_labels()

    print(features[0])
    print(labels[0])



if __name__ == "__main__":
    main()