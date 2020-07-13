from typing import List, Any

import matplotlib.pyplot as plt
import seaborn as sns


class DifferentLengthException(Exception):
    pass


def plot_losses(loss_values: List, validation_loss_values: List) -> None:
    """
    Plot losses functions
    :param loss_values: list of training loss values
    :param validation_loss_values: list of validation loss values
    :return: None
    """
    sns.set(style="darkgrid")

    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    plt.plot(loss_values, "b-o", label="training loss")
    plt.plot(validation_loss_values, "r-o", label="validation loss")

    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


def check_integrity(features: Any, labels: Any, desc: str) -> None:
    """
    Checks whether features and labels have the same length
    :param features: features
    :param labels: labels
    :param desc: description of the dataset (train, test, and so on)
    :return: None
    """
    i = 0
    for x, y in zip(features, labels):
        if len(x) != len(y):
            raise DifferentLengthException(
                "features and labels must have the same length. "
                "Row: {} of {}.\nlen: {}, {}\nlen: {}, {}".format(i, desc, len(x), x, len(y), y))
        i += 1
