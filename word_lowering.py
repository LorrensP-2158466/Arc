from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer


class WordLowerer():
    def lower(word: str):
        return word

class WordStemmer(WordLowerer):
    stemmer: SnowballStemmer = SnowballStemmer("english", ignore_stopwords=True)


    def lower(self, word: str) -> str:
        return self.stemmer.stem(word)


class WordLemmer(WordLowerer):
    lemmatizer: WordNetLemmatizer = WordNetLemmatizer()
    def lower(self, word: str) -> str:
        return self.lemmatizer.lemmatize(word)
