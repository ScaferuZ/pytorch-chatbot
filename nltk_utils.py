from nltk.stem.porter import PorterStemmer
import nltk
import numpy as np
# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentece):
    return nltk.word_tokenize(sentece)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog =   [   0,       1,   0,     1,     0,       0,      0]
    """
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1

    return bag
