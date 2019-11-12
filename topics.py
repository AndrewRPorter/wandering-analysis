import pandas as pd
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer

stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r"\w+")

BATCH_ONE = "./data/responses_batch1_numbers_clean.csv"
BATCH_TWO = "./data/responses_batch2_numbers_clean.csv"
STOP_LIST = "./data/stoplist.txt"


def run():
    docs = get_docs()
    stop_words = load_stop_list()
    all_tokens = [tokenizer.tokenize(doc) for doc in docs]

    pruned_tokens = []
    for tokens in all_tokens:
        tokens = [token for token in tokens if token not in stop_words]
        pruned_tokens.append(tokens)

    stemmed_tokens = []
    for tokens in pruned_tokens:
        tokens = [token for token in tokens if not token == "think"]
        stemmed_tokens.append(tokens)

    dictionary = corpora.Dictionary(stemmed_tokens)
    corpus = [dictionary.doc2bow(text) for text in stemmed_tokens]

    topics = {}

    # for some reason, eadh iteration of LdaModel yields different results...
    for i in range(0, 5):
        ldamodel = models.ldamodel.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=50)
        temp_topics = ldamodel.print_topics(num_topics=5, num_words=1)
        for topic in temp_topics:
            topic = topic[1].replace('"', "").split("*")[1]

            if topic in topics:
                topics[topic] += 1
            else:
                topics[topic] = 1

    print(topics)


def get_docs():
    """Creates a new 'document' for each response in the descriptions"""
    data = []

    d_one = pd.read_csv(BATCH_ONE)
    d_two = pd.read_csv(BATCH_TWO)

    for _, row in d_one.iterrows():
        data.append(row.description.lower())

    for _, row in d_two.iterrows():
        data.append(row.description.lower())

    return data


def load_stop_list():
    """Converts stoplist text file into list of tokens"""
    stop_list = []
    with open(STOP_LIST, "r") as f:
        lines = f.readlines()
        stop_list = [word.strip() for word in lines]
    return stop_list


if __name__ == "__main__":
    run()
