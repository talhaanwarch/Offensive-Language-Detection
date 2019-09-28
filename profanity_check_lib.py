# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 19:41:27 2019

@author: Talha
"""

from profanity_check import predict, predict_prob
import nltk

import tweepy
import re
import pickle
import matplotlib.pyplot as plt
from tweepy import OAuthHandler
import string
#%%
# Please change with your own consumer key, consumer secret, access token and access secret
# Initializing the keys
consumer_key = 'wHx5eCv2ZsEcVyQxfhbLWIzli'
consumer_secret = 'CC9iDbEAcutvTAS8VPK2RUoQckjLmyTxJWng0Jlpcyy4qyMdoT' 
access_token = '146371344-2RknzB58CZIguOVFJsjIxEgD1VhuoJFSLsBH8RKd'
access_secret ='18HstL1A1USI4wB5FNA98eglcACEzZsdMGcHFKXzggnlY'

# Initializing the tokens
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
#%%

args = ['#ImranKhanVoiceOfKashmir'];
api = tweepy.API(auth,wait_on_rate_limit=True)
#%%

# Fetching the tweets
list_tweets = []

query = args[0]
if len(args) == 1:
    for status in tweepy.Cursor(api.search,q=query+" -filter:retweets",lang='en',result_type='recent').items(10000):
        list_tweets.append(status.text)


#import pandas
#df = pandas.DataFrame(data={"tweets": list_tweets})
#df.to_csv("UNGA19.csv", sep=',',index=False)
#%%

not_offensive  =0
offensive=0
#preprocess the tweets
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet=re.sub('@[^\s]+','',tweet)
    tweet=re.sub('#[^\s]+','',tweet)
    tweet=tweet.translate(str.maketrans('', '', string.punctuation))

    #predict this tf idf vector class
    try:
        predicted_class=predict_prob(nltk.word_tokenize(tweet)).max()
        #offensive -0
        #not offensive 1
        if predicted_class > 0.6:
                offensive += 1
        else:
                not_offensive += 1
    except:
        pass        
#%%

#plot the graph
plt.pie([offensive,not_offensive],labels=['offensive','not_offensive'],autopct='%1.1f%%', shadow=True)
plt.title('Number of tweets about UNGA having offensive words')
