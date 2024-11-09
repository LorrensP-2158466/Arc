from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import polars as pl
import nltk
import re
import ast
from pprint import pprint

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import euclidean


class Modeling:
    dataset_path: str
    #df: pl.DataFrame
    stop_words: set[str]
    data: list[dict[str, str]]
    stemmer: SnowballStemmer
    lemmatizer: WordNetLemmatizer
    def __init__(self, dataset_path) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.dataset_path = dataset_path
        self.stemmer = SnowballStemmer("english")
        self.lemmatizer = WordNetLemmatizer()
        with open(self.dataset_path) as f:
            self.data = [
                {'year': int(item['year']), 'title': item['title']} 
                for line in f if line 
                for item in [ast.literal_eval(line.strip())]
            ]
            #self.df = pl.DataFrame(data)
    
    def model(self):
        (voc, preprocessed_titles) = self.make_vocabulary()
        vectorizer = TfidfVectorizer(vocabulary=voc)
        tfidf_matrix = vectorizer.fit_transform(preprocessed_titles)
        kmeans = KMeans(n_clusters=10, random_state=69)
        kmeans.fit(tfidf_matrix)
        clusters = kmeans.predict(tfidf_matrix) 


        k = 4
        # labels/topics per cluster
        topics = []

        for i in range(kmeans.n_clusters):
            # value on index i maps to cluster j and index i is also index into data
            cluster_indices = np.where(clusters == i)[0] 
            cluster_tfidf_matrix = tfidf_matrix[cluster_indices]
            mean_tfidf_scores = np.mean(cluster_tfidf_matrix.toarray(), axis=0)
            
            # top k words based on scores
            top_k_indices = np.argsort(mean_tfidf_scores)[::-1][:k]
            
            top_k_words = [vectorizer.get_feature_names_out()[index] for index in top_k_indices]
            
            topics.append((i, top_k_words))
        publications_per_year_per_cluster = defaultdict(lambda: defaultdict(int))
        # Iterate over each document, assigning it to the correct cluster and counting the publications by year
        for idx, cluster_id in enumerate(clusters):
            publication_year = self.data[idx]['year']  # Get the year for the current document
            publications_per_year_per_cluster[cluster_id][publication_year] += 1



        fig, ax = plt.subplots(figsize=(10, 6))


        for cluster_id, year_count in publications_per_year_per_cluster.items():
            years = sorted(year_count.items())  # Sort years
            years, counts = zip(*years)
            # Plotting each cluster with a distinct color
            ax.plot(years, counts, label=f"Topics: {topics[cluster_id][1]}")
            #ax.set_xticks(years[::3])  # Every 5th year

        # Set plot labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.set_title('Publications Per Year for Each Cluster')

        # Adding a legend
        ax.legend()

        # Display the plot
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust the plot layout to prevent clipping
        plt.show()

    
    def derive_topics():
        pass
    
    def tf_idf(self, voc: set[str], to_index: list[list[str]]):
        pass

    def make_vocabulary(self) -> tuple[set[str], list[str]]:
        voc = set() # GEKOLONISEERD
        new_titles: list[str] = []
        for publication in self.data:
            lowered_title = publication["title"].lower()
            title = re.sub('[^a-z0-9]', ' ', lowered_title)
            tokenized = word_tokenize(title);
            removed_stopwords = [w for w in tokenized if w not in self.stop_words]
            pre_processed = [self.lemmatizer.lemmatize(word) for word in removed_stopwords]
            new_titles.append(" ".join(pre_processed))
            voc.update(pre_processed)
        return (voc, new_titles)