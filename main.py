import argparse
import ast

from modeling import Modeling
from bert_topic_modeling import Bert
import nltk

def setup():
    nltk.download('stopwords') 
    nltk.download('punkt') 
    nltk.download('punkt_tab')
    nltk.download('wordnet')

def argument_parser():
    parser = argparse.ArgumentParser(description="Arc: Topic modeling tool.")
    
    # Define the flag for BERT
    parser.add_argument('--bert', action='store_true', help="Use BERT for topic modeling.")
    
    # Define the argument for dataset path
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset for topic modeling.")

    return parser
    


def main():
    args = argument_parser().parse_args()
    setup();

    path = args.dataset_path
    if args.bert:
        model = Bert(path)
        model.model()
    else:
        model = Modeling("datasets/data_mining_publications.txt")
        model.model()





if __name__ == "__main__": 
    main()