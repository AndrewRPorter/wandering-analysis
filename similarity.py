import math
import re
from collections import Counter

import data_helper

WORD = re.compile(r"\w+")
topics = []

with open("data/topics.txt", "r") as f:
    topics = f.readlines()
    topics = [topic.strip() for topic in topics]


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


def get_closest(user: str, numbers=False):
    response_to_topics = []

    for response in data_helper.get_sorted_responses(user):
        closest = (-1, 9)
        for index, topic in enumerate(topics):

            if topic in response[0]:
                if numbers:
                    closest = (1, index+1)
                else:
                    closest = (1, topic)
                break

            text1 = topic
            text2 = response[0]

            word_vector_one = text_to_vector(text1)
            word_vector_two = text_to_vector(text2)

            cosine = get_cosine(word_vector_one, word_vector_two)

            if cosine == 0.0:
                if numbers:
                    closest = (-1, 6)  # no topic found
                else:
                    closest = (-1, "other")  # no topic found

            if cosine > closest[0]:
                if numbers:
                    closest = (cosine, index+1)
                else:
                    closest = (cosine, topic)

        response_to_topics.append((response, closest[1]))

    return response_to_topics


def run():
    closest = get_closest()
    print(closest)


if __name__ == "__main__":
    run()
