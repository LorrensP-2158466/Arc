import ast

from modeling import Modeling
from bert_topic_modeling import Bert
import nltk

def setup():
    nltk.download('stopwords') 
    nltk.download('punkt') 
    nltk.download('punkt_tab')
    nltk.download('wordnet')

def main():
    setup();
    # model = Modeling("datasets/data_mining_publications.txt")
    # model.model()
    model = Bert("datasets/data_mining_publications.txt")
    model.model()




if __name__ == "__main__": 
    main()