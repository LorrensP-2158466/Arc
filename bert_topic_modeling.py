from matplotlib import pyplot as plt
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
import bertopic
import pandas as pd
import ast
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


class Bert():
    dataset_name: str
    data: pd.DataFrame
    def __init__(self, dataset_path: str):
     with open(dataset_path) as f:
        self.dataset_name = dataset_path.split("/")[-1].split(".")[0]
        data = [
            {'year': int(item['year']), 'title': item['title']} 
            for line in f if line 
            for item in [ast.literal_eval(line.strip())]
        ]
        self.data = pd.DataFrame(data)

    def model(self):


        # Create our beautiful BERT
        representation_model = KeyBERTInspired()
        vectorizer_model= CountVectorizer(tokenizer=LemmaTokenizer()) 
        topic_model = bertopic.BERTopic(
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            nr_topics="auto"
        )

        # perform BERT on titles
        titles = self.data['title'].tolist()
        topic_model.fit(titles)
        
        topics, _ = topic_model.fit_transform(titles)

        self.data['topic'] = topics
        topic_trends = self.data.groupby(['year', 'topic']).size().reset_index(name='count')

        pivot_table = topic_trends.pivot(index='year', columns='topic', values='count').fillna(0)
        
        topic_labels = topic_model.generate_topic_labels(separator=", ", topic_prefix=False)

        plt.figure(figsize=(12, 6))
        for idx, topic in enumerate(pivot_table.columns):
            label = topic_labels[idx]
            plt.plot(pivot_table.index, pivot_table[topic], label=f"Topic: {label}")

        plt.title(self.dataset_name)
        plt.xlabel("Year")
        plt.ylabel("Number of Publications")
        plt.legend(title="Topic", loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid()
        plt.tight_layout()
        plt.show()
