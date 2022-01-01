#Basic Python Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import re
import time
import warnings
warnings.filterwarnings("ignore") 
#Natural Language Processing
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#Scikit-Learn (Machine Learning Library for Python)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import spacy 
from spacy import displacy



# Loading the csv files into dataframds
dev = pd.read_csv('dev_sent_emo.csv')
train = pd.read_csv('train_sent_emo.csv')
test = pd.read_csv('test_sent_emo.csv')
#dev.head()
#train.head()
#test.head()
# We will work with only two dataframes- one for training and other for testing
train_dev = pd.concat([train,dev])
train_dev.head()
# Since we have concatenated the dataframes, resetting the indexes of the new one
train_dev.reset_index(inplace=True,drop=True)
# We will need only two columns for our sentiment analysis
sent = train_dev[['Utterance','Sentiment']]



# For our ML model to understand we will map the sentiments to numbers
def custom_encoder(df):
    df.replace(to_replace ="positive", value = 1, inplace=True)
    df.replace(to_replace ="neutral", value = 0, inplace=True)
    df.replace(to_replace ="negative", value = -1, inplace=True)
#Calling the funtion to encode the sentiments column
custom_encoder(sent['Sentiment'])
#Creating an object of WordNetLemmatizer
lm = WordNetLemmatizer()


# Function to preprocess text column to remove any characters other than alphabets, lemmatize the text, convert to lowercase
def data_preprocessing(text_col):
    corpus = []
    for row in text_col:
        new_row = row[0].capitalize() + row[1:].lower()
        new_row = re.sub('[^a-zA-Z]',' ',str(new_row)).split()
        new_row = [lm.lemmatize(word) for word in new_row if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_row))
    return corpus
transformed = data_preprocessing(sent['Utterance'])


# In the above data preprocessing step there were some empty strings. 
#Creating a dataframe to deal and remove those empty strings
tr_df = pd.DataFrame(zip(transformed,sent['Sentiment']), columns= ['Utterance','Sentiment'])
tr_df.head()
#Function to get the indices of the empty strings
def get_idx(df):   
    indexes = []
    for i,dialogue in enumerate(df):
        if len(dialogue) == 0:
            empty = df.index
            #print(empty)
            indexes.append(i)
    return indexes
empty_idx = get_idx(tr_df['Utterance'])

#Dropping the empty strings
clean_df = tr_df.drop(empty_idx)
clean_df.reset_index()

text = clean_df['Utterance'].to_list()
#text


# 1. NER Using NLTK 
words= []
for line in text:
    words.append(nltk.word_tokenize(line))
#words
#Converting the words which is list of list to a flat list of words
flat_word = [item for sublist in words for item in sublist]

#flat_word
#Part of speech tagging
pos_tags = nltk.pos_tag(flat_word)
#pos_tags

### Checking only whether it is Named Entity of not
chunks = nltk.ne_chunk(pos_tags, binary=False) #either NE or not NE
for chunk in chunks[:20]:
    print(chunk)
#Fetching NE and their labels
entities =[]
labels =[]
for chunk in chunks:
    if hasattr(chunk,'label'):
        #print(chunk)
        entities.append(' '.join(c[0] for c in chunk))
        labels.append(chunk.label())
        
entities_labels = list(set(zip(entities, labels)))
entities_df = pd.DataFrame(entities_labels, columns=["Entities","Labels"])
print(entities_df.sample(50))




# 2. NER Using Spacy
# Load SpaCy model
#nlp = spacy.load("en_core_web_sm")      # Small model
#nlp = spacy.load("en_core_web_md")      # Medium model
nlp = spacy.load("en_core_web_lg")      # Large model
#Fetching entities and their specific labels

entities = []
labels = []
position_start = []
position_end = []
for line in text:
    doc = nlp(line)

    for ent in doc.ents:
        entities.append(ent)
        labels.append(ent.label_)
    
ent_df = pd.DataFrame({'Entities':entities,'Labels':labels})
#pd.set_option("display.max_rows", 50)
print(ent_df.sample(50))



# Checking top 10 Entities 
plt.figure(figsize = (10,8))
plt.xlabel("Named Entities")
plt.ylabel("Count per Entity")
plt.title("Counts vs Named Entities")
plt.bar(ent_df['Labels'].value_counts().index[:10],ent_df['Labels'].value_counts().iloc[:10])
plt.show()
