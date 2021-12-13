#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 17:18:58 2021

@author: nihz415
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
%matplotlib inline
import seaborn as sns
import string
import warnings 
from scipy.special                   import expit
from sklearn                         import linear_model
from sklearn.decomposition           import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors               import KNeighborsClassifier
from sklearn.metrics                 import confusion_matrix
from sklearn.metrics                 import classification_report
%pip install wordcloud
import nltk
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from wordcloud import WordCloud


#step TWO:CLEAN DATA
##test and drop duplicates
dta.duplicated()
dta=dta.drop_duplicates()
dta.duplicated(['Text'])
dta=dta.drop_duplicates(['Text'])
##clean
def clean_text(text):
    text=re.sub(r'@[A-Za-z0-9]+',' ' ,text)
    text = re.sub(r'https?:\/\/.*\/\w*',' ',text)
    text = re.sub(r'[^a-zA-Z#]',' ',text)
    text = re.sub(r'#',' ',text)
    text = re.sub(r'RT[\s]+',' ',text)
    return text

dta['tidy_tweet'] = dta['Text'].astype('U').apply(clean_text)  
dta['tidy_tweet']=dta['tidy_tweet'].str.lower()   
dta['tidy_tweet'] = dta['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("n't"," not")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("I'm","I am")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("'ll"," will")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("It's","It is")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("it's","It is")
dta['tidy_tweet'] = dta['tidy_tweet'].str.replace("that's","that is")
##drop stopwords
stop = stopwords.words('english')
additional=['music core','music video','music bank','music','bank','core','award','nct','video','show','dream','127','wayv','nct','magic','carpet','ride','thwin']
stop+=additional
dta['tidy_tweet'] = dta['tidy_tweet'].str.split()
dta['tidy_tweet']=dta['tidy_tweet'].apply(lambda x:' '.join([item for item in x if item not in stop]))  
######
#classification
#find this part in clf.py
####
#futher analysis
dta_final.columns
len(dta_final)
dta_final.head()
#part 2:Discription
##overall count
dta_final['date']=dta_final['Datetime_x'].str.split(' ',expand=True)[0]
dta_final['date'] = pd.to_datetime(dta_final['date'])
count_in_day = dta_final.groupby(['date'],as_index=False).count()
#plot
date_format = mpl.dates.DateFormatter("%m-%d")
ax = plt.gca()
ax.xaxis.set_major_formatter(date_format)
plt.plot(count_in_day['date'], count_in_day['score'])
plt.axis('tight')
##After October 21, the degree of discussion dropped rapidly, so I deleted the content after October 20
dta1=dta_final[dta_final['date'].isin(pd.date_range('2021-08-20','2021-10-20'))]
##overall count again
count_in_day = dta1.groupby(['date'],as_index=False).count()
count_in_day['date'] = pd.to_datetime(count_in_day['date'])
#plot
date_format = mpl.dates.DateFormatter("%m-%d")
ax = plt.gca()
ax.xaxis.set_major_formatter(date_format)
plt.plot(count_in_day['date'], count_in_day['score'])
plt.axis('tight')
plt.show()
##mostly mentioned words
corpus= dta1['tidy_tweet_x'].to_list()   
vectorizer_count     = CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X                    = vectorizer_count.fit_transform(dta1['tidy_tweet_x'].values.astype('U'))
features_frequency   = pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape
features_frequency_sorted=features_frequency.sort_values('feature_frequency',ascending=False)
features_frequency_sorted.head(50)
sns.barplot(x="feature", y="feature_frequency", data=features_frequency.sort_values(by='feature_frequency',ascending=False).head(10))
plt.xticks(rotation=270)
plt.show()
#wordcloud
def worldcloudplot(features_frequency):
    wc1=features_frequency['feature'].tolist()
    wc2=features_frequency['feature_frequency'].tolist()
    wc= dict(zip(wc1,wc2))
    wordcloud = WordCloud(background_color='WHITE', height = 600, width = 800)
    wordcloud.generate_from_frequencies(frequencies=wc)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt.show()
worldcloudplot(features_frequency)
##mostly mentioned members
members = ['taeil','johnny','taeyong','yuta','doyoung','jaehyun','jungwoo','mark','haechan']
members_mentioned=features_frequency[features_frequency['feature'].isin(members)]
members_mentioned_sorted=members_mentioned.sort_values('feature_frequency',ascending=False)
members_mentioned_sorted
#attitude overall
##plot but it takes too long time
count_attitude=dta1.groupby(['predict'],as_index=False).count()
#timeseries of negative and positive
neg = dta1[dta1['predict']==1]
pos = dta1[dta1['predict']==3]
neu = dta1[dta1['predict']==2]
 
count_in_day_neg = neg.groupby(['date'],as_index=False).count()
count_in_day_neg['date'] = pd.to_datetime(count_in_day_neg['date'])
count_in_day_pos = pos.groupby(['date'],as_index=False).count()
count_in_day_pos['date'] = pd.to_datetime(count_in_day_pos['date'])
date_format = mpl.dates.DateFormatter("%m-%d")
ax = plt.gca()
ax.xaxis.set_major_formatter(date_format)
plt.plot(count_in_day_neg['date'], count_in_day_neg['predict'],label=neg)
plt.plot(count_in_day_pos['date'], count_in_day_pos['predict'],label=pos)
plt.axis('tight')

#positive part
##feature freq
corpus= pos['tidy_tweet_x'].to_list()   
vectorizer_count= CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X= vectorizer_count.fit_transform(pos['tidy_tweet_x'].values.astype('U'))
features_frequency_pos= pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape
features_frequency_pos_sorted=features_frequency_pos.sort_values('feature_frequency',ascending=False)
features_frequency_pos_sorted.head(50)
plt.xticks(rotation=270)
sns.barplot(x="feature", y="feature_frequency", data=features_frequency_pos.sort_values(by='feature_frequency',ascending=False).head(10))
##wc
worldcloudplot(features_frequency_pos)
##members mentioned
members_mentioned_pos=features_frequency_pos[features_frequency_pos['feature'].isin(members)]
members_mentioned_pos_sorted=members_mentioned_pos.sort_values('feature_frequency',ascending=False)
members_mentioned_pos_sorted
#negative part
##feature freq
corpus= neg['tidy_tweet_x'].to_list()   
vectorizer_count=CountVectorizer(lowercase   = True,ngram_range = (1,1),max_df      = 0.99,min_df      = 0.001);
X=vectorizer_count.fit_transform(neg['tidy_tweet_x'].values.astype('U'))
features_frequency_neg=pd.DataFrame({'feature'           : vectorizer_count.get_feature_names(),'feature_frequency' : X.toarray().sum(axis=0)})
X.shape
features_frequency_neg_sorted=features_frequency_neg.sort_values('feature_frequency',ascending=False)
features_frequency_neg_sorted.head(50)
plt.xticks(rotation=270)
sns.barplot(x="feature", y="feature_frequency", data=features_frequency_neg.sort_values(by='feature_frequency',ascending=False).head(10))
##wc
worldcloudplot(features_frequency_neg)
##members mentioned
members_mentioned_neg=features_frequency_neg[features_frequency_neg['feature'].isin(members)]
members_mentioned_neg_sorted=members_mentioned_neg.sort_values('feature_frequency',ascending=False)
members_mentioned_neg_sorted


