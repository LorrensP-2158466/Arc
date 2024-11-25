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
    
    parser.add_argument('--bert', action='store_true', help="Use BERT for topic modeling.")
    
    parser.add_argument('--dataset-path', type=str, required=True, help="Path to the dataset for topic modeling.")

    parser.add_argument('--nr-of-topics', type=int, default=6, help="Number of topics to find (clusters to use)")

    parser.add_argument('--repr-nr-topics', type=int, default=5, help="Number of topics to represent in the plot based on total Publications op topic")

    parser.add_argument('--use-stemming', action='store_true', help="Wether to use Stemming or Lemmatization")

    parser.add_argument('--interval', type=int, default=5, help="Smooth out the curves by creating year intervals on the ranges (don't do 1 please)")

    parser.add_argument('--save-plot', type=str,  help="Save the plot to this path instead of showing it")

    return parser
    


def main():
    #setup();
    args = argument_parser().parse_args()

    path = args.dataset_path
    if args.bert:
        model = Bert(path)
        model.model(args.nr_of_topics, args.repr_nr_topics, args.use_stemming, args.interval, args.save_plot)
    else:
        model = Modeling(path,  args.use_stemming)
        model.model(args.nr_of_topics, args.repr_nr_topics, args.interval, args.save_plot)





if __name__ == "__main__": 
    main()