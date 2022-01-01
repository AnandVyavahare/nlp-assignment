#Basic Python Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import re
import time
import warnings
import seaborn as sns
warnings.filterwarnings("ignore") 
#Natural Language Processing
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#nltk.download('stopwords')
#Scikit-Learn (Machine Learning Library for Python)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
#Evaluation Metrics
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,classification_report,plot_confusion_matrix



# Loading the csv files into dataframds
dev = pd.read_csv('dev_sent_emo.csv')
train = pd.read_csv('train_sent_emo.csv')
test = pd.read_csv('test_sent_emo.csv')
#dev.head()
#train.head()
test.head()

# We will work with only two dataframes- one for training and other for testing
train_dev = pd.concat([train,dev])
train_dev
# Since we have concatenated the dataframes, resetting the indexes of the new one
train_dev.reset_index(inplace=True,drop=True)
# We will need only two columns for our sentiment analysis
sent = train_dev[['Utterance','Sentiment']]


# Data Preprocessing
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
        new_row = re.sub('[^a-zA-Z]',' ',str(row)).lower().split()
        new_row = [lm.lemmatize(word) for word in new_row if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_row))
    return corpus
transformed = data_preprocessing(sent['Utterance'])
#transformed

# In the above data preprocessing step there were some empty strings. 
#Creating a dataframe to deal and remove those empty strings
tr_df = pd.DataFrame(zip(transformed,sent['Sentiment']), columns= ['Utterance','Sentiment'])
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
#clean_df['Sentiment'].value_counts()



## Using TFIDF
# Creating an instance of TFIDF
vectorizer = TfidfVectorizer(use_idf=True, norm= 'l1', ngram_range=(2,2))
traindata = vectorizer.fit_transform(text)
#len(vectorizer.get_feature_names())
#traindata



# Parameters for Gridsearch
parameters = {'n_estimators': [100,150,200],
             'max_depth': [30,40,50],
             'min_samples_split': [15,20,30],
             'min_samples_leaf': [2, 5]
             }


grid_search = GridSearchCV(RandomForestClassifier(),parameters,cv=5,return_train_score=True,n_jobs=-1)
grid_search.fit(traindata,clean_df['Sentiment'])
grid_search.best_params_


# Implementing Random Forest
rfc = RandomForestClassifier(max_depth=grid_search.best_params_['max_depth'],
                             n_estimators=grid_search.best_params_['n_estimators'],
                             min_samples_split=grid_search.best_params_['min_samples_split'],
                             min_samples_leaf=grid_search.best_params_['min_samples_leaf']
                            )
rfc.fit(traindata,clean_df['Sentiment'])

#from sklearn.naive_bayes import MultinomialNB

#nb = MultinomialNB(alpha=0.75)
#nb.fit(traindata,clean_df['Sentiment'])


# Working with test dataset
test_df = test[['Utterance','Sentiment']]
X_test,y_test = test_df['Utterance'],test_df['Sentiment']

#Pre-processing text data
test_transformed = data_preprocessing(X_test)
test_tr_df = pd.DataFrame(zip(test_transformed,test['Sentiment']),columns= ['Utterance','Sentiment'])
empty_test_idx = get_idx(test_tr_df['Utterance'])
clean_test_df = test_tr_df.drop(empty_test_idx)

#Convert text data into vectors
test_text = clean_test_df['Utterance'].to_list()
testdata = vectorizer.transform(test_text)
X_test,y_test = clean_test_df['Utterance'],clean_test_df['Sentiment']

#Encode the labels into three classes 0, 1, and -1
custom_encoder(y_test)
#predict the target
predictions = rfc.predict(testdata)



### Evaluating the results
# Calculating Accuracy, Precision, and Recall scores
#plt.figure(figsize = (12,8))
#plot_confusion_matrix(y_test,predictions)
acc_score = round(accuracy_score(y_test,predictions),3)
pre_score = round(precision_score(y_test,predictions,average='weighted'),2)
rec_score = round(recall_score(y_test,predictions,average='weighted'),2)

print('Accuracy_score: ',acc_score)
print('Precision_score: ',pre_score )
print('Recall_score: ',rec_score)
print("-"*55)

# Generating Classification Report
cr = classification_report(y_test,predictions)
print(cr)

#Confusion Matrices for Test
cm_test_rf = confusion_matrix(y_test,predictions)
cm_test_rf_df = pd.DataFrame(cm_test_rf)

#Plotting Heatmap for Test Confusion Matrix
sns.heatmap(cm_test_rf_df, fmt='g', annot = True, cmap ='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confustion Matrix of test')

