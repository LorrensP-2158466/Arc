
from pprint import pprint
from matplotlib import pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import ast

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn_extra.cluster import KMedoids

from word_lowering import WordLowerer, WordLemmer, WordStemmer



class Modeling:
    dataset_path: str
    dataset_name: str
    #df: pl.DataFrame
    stop_words: set[str]
    data: pd.DataFrame
    word_lowerer: WordLowerer
    # go from lowered word to original (were it came from)
    un_lowerer: dict[str, str]
    
    def __init__(self, dataset_path, use_stemmer: bool = False) -> None:
        self.dataset_name = dataset_path.split("/")[-1].split(".")[0]
        with open("stopwords_kaggle.txt") as stop_words_file:
            self.stop_words = set(map(lambda line: line.strip(), stop_words_file))
        self.dataset_path = dataset_path
        self.un_lowerer = {}
        self.found_stopwords = set()
        if use_stemmer:
            self.word_lowerer = WordStemmer()
        else:
            self.word_lowerer = WordLemmer()

        with open(self.dataset_path) as f:
            data = [
                {'year': int(item['year']), 'title': item['title']} 
                for line in f if line 
                for item in [ast.literal_eval(line.strip())]
            ]
            self.data = pd.DataFrame(data)

        
    def model(self, nr_clusters: int = 6):
        self.make_vocabulary()
        pprint(self.found_stopwords)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.data["preprocessed_titles"])

        kmedoids = KMeans(n_clusters=nr_clusters, random_state=2158466)
        labels = kmedoids.fit(tfidf_matrix).labels_

        self.data['topic'] = labels
        topic_trends = self.data.groupby(['year', 'topic']).size().reset_index(name='count')
        pivot_table = topic_trends.pivot(index='year', columns='topic', values='count').fillna(0)


        plt.figure(figsize=(12, 12))
        topic_labels = self.create_topic_labels(tfidf_matrix, 6, labels, vectorizer)
        for idx, topic in enumerate(pivot_table.columns):
            topic_label = topic_labels[idx]
            plt.plot(pivot_table.index, pivot_table[topic], label=f"Topic: {topic_label}")
        
        plt.title(f"Arc Topic flow in dataset: {self.dataset_name}")
        plt.xlabel("Year")
        plt.ylabel("Number of Publications")
        plt.legend(title="Topic", loc='center left', bbox_to_anchor=(1, 0))
        plt.tight_layout()
        plt.show()

    def create_topic_labels(self, matrix, n_clusters, labels, vectorizer):
        pprint(labels)
        k = 3  
        topics = []

        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) == 0:
                topics.append(f"Cluster {i}: No data")
                continue

            cluster_tfidf_matrix = matrix[cluster_indices]
            mean_tfidf_scores = np.mean(cluster_tfidf_matrix.toarray(), axis=0)

            top_k_indices = np.argsort(mean_tfidf_scores)[::-1][:k]
            top_k_words = [vectorizer.get_feature_names_out()[index] for index in top_k_indices]

            topics.append(" ".join(map(lambda w: self.un_lowerer[w], top_k_words)))

        return topics
    
    def preprocess_title(self, title):
        title = re.sub(r'\W+', ' ', title)
        title = re.sub(r'\d+', '', title)
        title = title.lower()
        preproc_title = ""
        for word in filter(lambda w: w not in self.stop_words, word_tokenize(title)):
            lowered = self.word_lowerer.lower(word);
            preproc_title += f" {lowered}"
            self.un_lowerer[lowered] = word
        return preproc_title


    def make_vocabulary(self):
        self.data["preprocessed_titles"] = self.data["title"].apply(lambda title: self.preprocess_title(title))