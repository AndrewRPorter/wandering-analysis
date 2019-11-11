import string

import pandas as pd

BATCH_ONE = "./data/responses_batch1_numbers_clean.csv"
BATCH_TWO = "./data/responses_batch2_numbers_clean.csv"
RESPONSES = "./data/responses.csv"
STOP_LIST = "./data/stoplist.txt"


def load_stop_list():
    stop_list = []
    with open(STOP_LIST, "r") as f:
        lines = f.readlines()
        stop_list = [word.strip() for word in lines]
    return stop_list


def replace_all(word, replacement):
    """Replaces puncation in word"""
    punctation = list(string.punctuation)
    punctation = [i for i in punctation if not i == "'"]

    for char in punctation:
        word = word.replace(char, replacement)

    return word


def prune(data):
    stop_list = load_stop_list()
    words = data.split()
    words = [replace_all(word, "") for word in words]
    words = [word.lower() for word in words if word]  # remove empty strings from list
    words = [word for word in words if word not in stop_list]
    return words


def get_data():
    return pd.read_csv(RESPONSES)


def get_sorted_responses(data: pd.DataFrame):
    responses = []
    current_user = None

    for index, row in data.iterrows():
        row = dict(row)
        if not current_user:
            current_user = row["users"]

        if row["users"] != current_user and str(row["users"]) != "nan":
            break  # we only want to process one user for right now

        if len(row["response"].strip()) > 1:  # only want thought data
            responses.append((row["response"], row["responseTimestamp"]))

    sorted_response = sorted(responses, key=lambda x: x[1])
    return sorted_response
