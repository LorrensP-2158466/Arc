from matplotlib import pyplot as plt
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
import bertopic
import pandas as pd
import ast
from nltk import word_tokenize          
from nltk.corpus import stopwords
from word_lowering import WordLemmer, WordStemmer, WordLowerer


class Tokenizer:
    lowerer: WordLowerer
    def __init__(self, lowerer: WordLowerer):
        self.lowerer = lowerer
    def __call__(self, doc):
        return [self.lowerer.lower(t) for t in word_tokenize(doc)]


class Bert():
    dataset_name: str
    data: pd.DataFrame
    def __init__(self, dataset_path: str):
     with open(dataset_path) as f:
        self.dataset_name = dataset_path.split("/")[-1].split(".")[0]
        with open("stopwords_kaggle.txt") as stop_words_file:
            self.stop_words = list(map(lambda line: line.strip(), stop_words_file))
        data = [
            {'year': int(item['year']), 'title': item['title']} 
            for line in f if line
            for item in [ast.literal_eval(line.strip())]
        ]
        self.data = pd.DataFrame(data)

    def model(self, nr_topics: int = 6, use_stemmer: bool = False):
        # Create our beautiful BERT
        tokenizer = 0
        if use_stemmer:
            tokenizer = Tokenizer(WordStemmer())
        else:
            tokenizer = Tokenizer(WordLemmer())
        

        representation_model = KeyBERTInspired()
        vectorizer_model= CountVectorizer(tokenizer=tokenizer, stop_words=self.stop_words) 
        topic_model = bertopic.BERTopic(
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            nr_topics=nr_topics
        )

        # perform BERT on titles
        titles = self.data['title'].tolist()
        topic_model.fit(titles)
        
        topics, _ = topic_model.fit_transform(titles)

        self.data['topic'] = topics
        topic_trends = self.data.groupby(['year', 'topic']).size().reset_index(name='count')

        pivot_table = topic_trends.pivot(index='year', columns='topic', values='count').fillna(0)
        
        topic_labels = topic_model.generate_topic_labels(separator=" ", topic_prefix=False)

        plt.figure(figsize=(12, 12))
        for idx, topic in enumerate(pivot_table.columns):
            label = topic_labels[idx]
            plt.plot(pivot_table.index, pivot_table[topic], label=f"Topic: {label}")

        plt.title(f"BERT Topic flow in dataset: {self.dataset_name}")
        plt.xlabel("Year")
        plt.ylabel("Number of Publications")
        plt.legend(title="Topic", loc='center left', bbox_to_anchor=(1, 0))
        plt.tight_layout()
        plt.show()
