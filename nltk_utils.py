from nltk.stem.porter import PorterStemmer
import nltk
# nltk.download('punkt')
stemmer = PorterStemmer()


def tokenize(sentece):
    return nltk.word_tokenize(sentece)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    pass
