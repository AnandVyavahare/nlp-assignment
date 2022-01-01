#Basic Python Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import re
import time
#Natural Language Processing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#Scikit-Learn (Machine Learning Library for Python)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import LatentDirichletAllocation
dev = pd.read_csv('dev_sent_emo.csv')
train = pd.read_csv('train_sent_emo.csv')
test = pd.read_csv('test_sent_emo.csv')
#dev.head()
#train.head()
test.head()
train_dev = pd.concat([train,dev])
train_dev
train_dev.reset_index(inplace=True,drop=True)
sent = train_dev[['Utterance','Sentiment']]
def custom_encoder(df):
    df.replace(to_replace ="positive", value = 1, inplace=True)
    df.replace(to_replace ="neutral", value = 0, inplace=True)
    df.replace(to_replace ="negative", value = -1, inplace=True)
custom_encoder(sent['Sentiment'])
#Creating an object of WordNetLemmatizer
lm = WordNetLemmatizer()
def data_preprocessing(text_col):
    corpus = []
    for row in text_col:
        new_row = re.sub('[^a-zA-Z]',' ',str(row)).lower().split()
        new_row = [lm.lemmatize(word) for word in new_row if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_row))
    return corpus
transformed = data_preprocessing(sent['Utterance'])
#transformed
tr_df = pd.DataFrame(zip(transformed,sent['Sentiment']), columns= ['Utterance','Sentiment'])
def get_idx(df):   
    indexes = []
    for i,dialogue in enumerate(df):
        if len(dialogue) == 0:
            empty = df.index
            #print(empty)
            indexes.append(i)
    return indexes
empty_idx = get_idx(tr_df['Utterance'])
clean_df = tr_df.drop(empty_idx)
clean_df.reset_index()
text = clean_df['Utterance'].to_list()
text
#clean_df['Sentiment'].value_counts()
## Topic Modeling with LDA in Sklearn
tfidf = TfidfVectorizer(use_idf=True, norm= 'l1')
traindata = tfidf.fit_transform(text)
#len(tfidf.get_feature_names())
type(traindata)
### Creating the vocab to represent all the corpus
vocab_tfidf = tfidf.get_feature_names()
len(vocab_tfidf)
### Implementing LDA
# Instantiating LDA
lda_model = LatentDirichletAllocation(n_components=6, max_iter = 20, random_state = 42, batch_size= 500)

# Fitting and transforming model on our tfidf vectorizer to get topics
x_topics = lda_model.fit_transform(traindata)

# Checking topic distribution
topic_words = lda_model.components_
topic_words.shape
topic_words
# defining the number of words to print in every topic
n_top_words = 6

for i, topic_dist in enumerate(topic_words):
    #sorting the indices so in the topic_words array
    sorted_topic_dist = np.argsort(topic_dist)
    
    #Fetching the actual words for the sorted indices above
    actual_topic_words = np.array(vocab_tfidf)[sorted_topic_dist]
    
    #Showing top n_top_words per topic
    n_top_topic_words = actual_topic_words[:-n_top_words:-1]
    print ("Topic", str(i+1), n_top_topic_words)
actual_topic_words
# To view what topics are assigned to the douments:

doc_topic = lda_model.transform(traindata)  

# iterating over ever value till the end value
for n in range(20):
    
    # argmax() gives maximum index value
    topic_doc = doc_topic[n].argmax()
    
    # document is n+1  
    print ("Document", n+1, " -- Topic:" ,topic_doc)
for i in range(20):
    print("Document", i+1,text[i])

