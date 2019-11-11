import pandas as pd
from matplotlib import pyplot as plt

from data_helper import get_data, get_sorted_responses


def run():
    data = get_data()
    sorted_responses = get_sorted_responses(data)
    plot(sorted_responses)


def plot(data):
    responses = [i[0] for i in data]

    fig, ax = plt.subplots()
    plt.xticks, plt.yticks = range(0, len(responses)), range(0, len(responses))
    ax.set_xticklabels(range(0, len(responses)))
    ax.set_yticklabels(range(0, len(responses)))
    plt.plot(responses, responses)
    plt.scatter(responses, responses)
    plt.show()


if __name__ == "__main__":
    run()
