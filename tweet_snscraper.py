#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 21:59:57 2021

@author: nihz415
"""

pip install snscrape
pip install git+https://github.com/JustAnotherArchivist/snscrape.git


import snscrape.modules.twitter as sntwitter
import pandas
from pandas import DataFrame

# Creating list to append tweet data to
tweets=[]
# Using TwitterSearchScraper to scrape data and append tweets to list
for i,tweet in enumerate(sntwitter.TwitterSearchScraper('#NCT127_Sticker lang:en ').get_items()):
    tweets.append([tweet.date, tweet.id, tweet.content, tweet.user.username])
    
# Creating a dataframe from the tweets list above
len(tweets)
tweets[-1]
tweets_df = DataFrame(tweets, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
tweets_df
outputpath='/Users/nihz415/Desktop/final project/datacanbeused/nctSTICKER.csv'
tweets_df.to_csv(outputpath,index=True,header=True)
