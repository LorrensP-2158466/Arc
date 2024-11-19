from collections import defaultdict
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
import ast

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn_extra.cluster import KMedoids


class Modeling:
    dataset_path: str
    #df: pl.DataFrame
    stop_words: set[str]
    data: pd.DataFrame
    stemmer: SnowballStemmer
    lemmatizer: WordNetLemmatizer
    def __init__(self, dataset_path) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.dataset_path = dataset_path
        self.stemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.lemmatizer = WordNetLemmatizer()
        with open(self.dataset_path) as f:
            data = [
                {'year': int(item['year']), 'title': item['title']} 
                for line in f if line 
                for item in [ast.literal_eval(line.strip())]
            ]
            self.data = pd.DataFrame(data)
    
    def model(self):
        self.make_vocabulary()
        vectorizer = TfidfVectorizer(max_features=2500)
        tfidf_matrix = vectorizer.fit_transform(self.data["preprocessed_titles"])

        kmedoids = KMedoids(n_clusters=6, random_state=2158466, metric='cosine')
        labels = kmedoids.fit(tfidf_matrix).labels_

        self.data['topic'] = labels
        topic_trends = self.data.groupby(['year', 'topic']).size().reset_index(name='count')
        pivot_table = topic_trends.pivot(index='year', columns='topic', values='count').fillna(0)
        plt.figure(figsize=(12, 6))
        topic_labels = self.create_topic_labels(tfidf_matrix, 6, labels, vectorizer)

        for idx, topic in enumerate(pivot_table.columns):
            topic_label = topic_labels[idx]
            plt.plot(pivot_table.index, pivot_table[topic], label=f"Topic: {topic_label}")
        
        plt.title("u mama")
        plt.xlabel("Year")
        plt.ylabel("Number of Publications")
        plt.legend(title="Topic", loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid()
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

            topics.append(", ".join(top_k_words))

        return topics
    
    def preprocess_title(self, title):
        title = re.sub(r'\W+', ' ', title)
        title = re.sub(r'\d+', '', title)
        title = title.lower()
        title = ' '.join([self.stemmer.stem(word) for word in word_tokenize(title) if word not in self.stop_words])
        return title


    def make_vocabulary(self):
        self.data["preprocessed_titles"] = self.data["title"].apply(lambda title: self.preprocess_title(title))