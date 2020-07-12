from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


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
