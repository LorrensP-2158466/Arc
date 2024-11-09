import ast

from modeling import Modeling
import nltk

def setup():
    nltk.download('stopwords') 
    nltk.download('punkt') 
    nltk.download('punkt_tab')
    nltk.download('wordnet')

def main():
    setup();
    model = Modeling("datasets/database_systems_publications.txt")
    model.model()




if __name__ == "__main__": 
    main()