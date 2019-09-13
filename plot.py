"""
Violin plot code adjusted from
https://matplotlib.org/examples/statistics/customized_violin_demo.html
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

BATCH_ONE = "./data/responses_batch1_numbers_clean.csv"
BATCH_TWO = "./data/responses_batch2_numbers_clean.csv"


def run():
    data = get_data()
    plot(data)


def get_data():
    data = []

    d_one = pd.read_csv(BATCH_ONE)
    d_two = pd.read_csv(BATCH_TWO)

    for _, row in d_one.iterrows():
        row.off_task = 1 if row.off_task.strip() == "y" else 0  # map qualitative values to numbers
        data.append((row.off_task, row.wandering, row.description))

    for _, row in d_two.iterrows():
        row.off_task = 1 if row.off_task.strip() == "y" else 0  # map qualitative values to numbers
        data.append((row.off_task, row.wandering, row.description))

    return data


def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction="out")
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


def plot(data):
    on_task, off_task = [], []

    for val in data:
        if val[0] == 0:  # on task
            on_task.append(val[1])
        else:
            off_task.append(val[1])

    fig, ax = plt.subplots()
    plt.title("Mind Wandering Bins")
    plt.ylabel("Wandering (1-6)")
    plt.violinplot([on_task, off_task])

    labels = ["On Task", "Off Task"]
    set_axis_style(ax, labels)

    plt.yticks([1, 2, 3, 4, 5, 6])
    plt.savefig("./output/figure.png")
    plt.show()


if __name__ == "__main__":
    run()
