from matplotlib import pyplot as plt
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
        with open("stopwords_kaggle.txt") as stop_words_file:
            self.stop_words = list(map(lambda line: line.strip(), stop_words_file))
        with open(dataset_path, "r", encoding="utf-8") as f:
            self.dataset_name = dataset_path.split("/")[-1].split(".")[0]

            data = [
                {'year': int(item['year']), 'title': item['title']} 
                for line in f if line
                for item in [ast.literal_eval(line.strip())]
            ]
            self.data = pd.DataFrame(data)

    def model(self, nr_topics: int = 6,repr_topics: int = 5, use_stemmer: bool = False, interval: int = 5):
        # Create our beautiful BERT
        tokenizer = 0
        if use_stemmer:
            tokenizer = Tokenizer(WordStemmer())
        else:
            tokenizer = Tokenizer(WordLemmer())
        


        vectorizer_model= CountVectorizer(tokenizer=tokenizer, stop_words=self.stop_words) 
        topic_model = bertopic.BERTopic(
            vectorizer_model=vectorizer_model,
            nr_topics=nr_topics
        )

        # perform BERT on titles
        titles = self.data['title'].tolist()
        topics, _ = topic_model.fit_transform(titles)
        
        topic_labels = topic_model.generate_topic_labels(separator=" ", topic_prefix=False)

        self.data['topic'] = topics

        topic_counts = self.data[self.data['topic'] != -1]['topic'].value_counts()
        top_topics = topic_counts.nlargest(repr_topics).index.tolist()

        topic_trends = self.data.groupby(['year', 'topic']).size().reset_index(name='count')
        topic_trends = topic_trends[topic_trends['topic'].isin(top_topics) & (topic_trends['year'] < 2024)]

        # Create our intervals
        topic_trends['year_interval'] = (topic_trends['year'] // interval) * interval
        interval_trends = topic_trends.groupby(['year_interval', 'topic'])['count'].sum().reset_index()

        pivot_table = interval_trends.pivot(index='year_interval', columns='topic', values='count').fillna(0)



        print(topic_labels)
        plt.figure(figsize=(12, 6))
        for topic in pivot_table.columns:
            label = topic_labels[topic]
            plt.plot(pivot_table.index, pivot_table[topic], 
                    label=f"{topic} {label}")  

        plt.xticks(pivot_table.index, [f'{year}-{year+interval-1}' for year in pivot_table.index])
        plt.xticks(rotation=45)

        plt.title(f"Top Topics by {interval}-Year Intervals in {self.dataset_name}")
        plt.xlabel("Year Range")
        plt.ylabel("Number of Publications")
        plt.legend(title="Topics", loc='center left', bbox_to_anchor=(1.05, 0.5))
        plt.tight_layout()
        plt.show()
