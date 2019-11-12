import csv
import os
import numpy as np

import pylab
from pyunicorn.timeseries import RecurrencePlot, RecurrenceNetwork

from similarity import get_closest
import data_helper

users = data_helper.get_users()
_participant_file = "output/participants.csv"

with open(_participant_file, "w") as f:
    writer = csv.writer(f, delimiter=",", quotechar="'", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["user", "#responses", "recurrence_rate", "determinism", "laminarity"])

    for user in users:
        time_series = [float(i[1]) for i in get_closest(user, numbers=True)]
        time_series = np.array(time_series)  # need to convert for RQA analysis with pyunicorn

        #  Settings for the embedding
        DIM = 1  # Embedding dimension
        TAU = 0  # Embedding delay

        #  Settings for the recurrence plot
        EPS = 0.05  # Fixed threshold
        METRIC = "euclidean"  # ("manhattan","euclidean","supremum")

        pylab.plot(time_series, "r")
        pylab.xlabel("$n$")
        pylab.ylabel("$x_n$")

        #  Generate a recurrence plot object with fixed recurrence threshold EPS
        rp = RecurrencePlot(time_series, dim=DIM, tau=TAU, metric=METRIC, normalize=False, threshold=EPS)

        #  Show the recurrence plot
        pylab.matshow(rp.recurrence_matrix())
        pylab.xlabel("$n$")
        pylab.ylabel("$n$")
        pylab.savefig(f"output/{user}_rqa.png")

        #  Calculate some standard RQA measures
        DET = rp.determinism(l_min=2)
        LAM = rp.laminarity(v_min=2)

        writer.writerow([user, len(time_series), rp.recurrence_rate(), DET, LAM])