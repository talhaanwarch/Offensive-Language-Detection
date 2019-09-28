# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:26:27 2019

@author: Talha
data source:https://sites.google.com/site/offensevalsharedtask/olid

"""


#import libraries
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

#%%import data
directory='D:/Datasets/text/OLIDv1.0/'
train=pd.read_csv(directory + 'olid-training-v1.0.tsv',sep="\t")
test=pd.read_csv(directory + 'testset-levela.tsv',sep="\t")
y_test=pd.read_csv(directory + 'labels-levela.csv',header=None).iloc[:,-1]
#%%
y_train=train['subtask_a']
train=train['tweet']
test=test['tweet']
y_train=pd.factorize(y_train)[0]
y_test=pd.factorize(y_test)[0]


#%%
# Creating the training corpus
corpus_train = []
for i in train:
    x=i.lower()
    x=x.replace('@user','')
    x=x.translate(str.maketrans('', '', string.punctuation))
    x = re.sub('[^A-Za-z]', ' ', x)
    corpus_train.append(x)    
# Creating the training corpus
corpus_test = []
for i in test:
    x=i.lower()
    x=x.replace('@user','')
    x=x.translate(str.maketrans('', '', string.punctuation))
    x = re.sub('[^A-Za-z]', ' ', x)
    corpus_test.append(x) 
#%%stemming and lematization
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer

porter=PorterStemmer()
lemmatizer=WordNetLemmatizer()
lancaster = LancasterStemmer()


train_corpus_stem=[]
for sentence in corpus_train:
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(lancaster.stem(lemmatizer.lemmatize(word)))
        stem_sentence.append(" ")
    train_corpus_stem.append( "".join(stem_sentence))


test_corpus_stem=[]
for sentence in corpus_test:
    token_words=word_tokenize(sentence)
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(lancaster.stem(lemmatizer.lemmatize(word)))
        stem_sentence.append(" ")
    test_corpus_stem.append( "".join(stem_sentence))
#%%    
    
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 100)
X_train = vectorizer.fit_transform(corpus_train).toarray()
X_test = vectorizer.fit_transform(corpus_test).toarray()

#%% bag of words

#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(max_features = 500,ngram_range=(1,1))
#X_train = vectorizer.fit_transform(corpus_train).toarray()
#X_test = vectorizer.fit_transform(corpus_test).toarray()

#%%
# Naive Bayes 
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
from sklearn.svm import SVC
classifier=SVC(C=10,gamma=0.1)
classifier.fit(X_train, y_train)
# Predict Class
y_pred = classifier.predict(X_test)
# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
#%% neural net 
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

model=Sequential()
model.add(Dense(32,input_dim=(100),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
optimizer=keras.optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,   epochs=30,  verbose=2, validation_data=(X_test, y_test), batch_size=10)

from keras import backend as K 
K.clear_session()
#%%Save model
# Saving our classifier
import pickle
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
# Saving the Tf-Idf model
with open('tfidfmodel.pickle','wb') as f:
    pickle.dump(vectorizer,f)
    
#%% grid search
from sklearn.model_selection import GridSearchCV
def svc_param_selection(X, y, nfolds):
    Cs = [ 0.1, 1,5, 5,10,20]
    gammas = [0.001, 0.01, 0.1,0.2,0.5, 1,1.5]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(svm.SVC(kernel='rbf'), param_grid, cv=nfolds,n_jobs=-1)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search.best_params_

grid=svc_param_selection(X_train,y_train,5)    