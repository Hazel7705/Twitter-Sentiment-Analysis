#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 16:30:55 2021

@author: nihz415
"""

import csv
from getpass import getpass
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome
from pandas import DataFrame
import datetime
import gc
import os
import pandas as pd
import re
import shelve
import time
import datetime
from selenium     import webdriver



path = '/Users/nihz415/Documents/学习/BUS256Marketing'
os.chdir(path)
driver = webdriver.Chrome('/Users/nihz415/Documents/学习/BUS256Marketing/chromedriver')
#login
driver.get('http://www.twitter.com/login')
username=driver.find_element_by_xpath('//input[@name="username"]')
username.send_keys('nihz415@outlook.com')
username.send_keys(Keys.RETURN)
password=driver.find_element_by_xpath('//input[@name="password"]')
password.send_keys('Ultimate415')
password.send_keys(Keys.RETURN)
#search
search=driver.find_element_by_xpath('//*[@id="react-root"]/div/div/div[2]/header/div/div/div/div[1]/div[2]/nav/a[2]')
search.click()
search_input=driver.find_element_by_xpath('//input[@aria-label="查询词条"]')
search_input.send_keys('nct music lang:en since:2021-7-1 until:2021-7-11')
search_input.send_keys(Keys.RETURN)
driver.find_element_by_link_text('最新').click()

#Scrape
cards=driver.find_elements_by_xpath('//article[@data-testid="tweet"]')



#function
def get_tweet_data(card):
    username=card.find_element_by_xpath('.//span').text
    twitterID=card.find_element_by_xpath('.//span[contains(text(),"@")]').text
    try:
        postdate=card.find_element_by_xpath('.//time').get_attribute('datetime')
    except NoSuchElementException:
        return
    content=card.find_element_by_xpath('.//div[2]/div[2]/div[2]').text
    responding=card.find_element_by_xpath('.//div[2]/div[2]/div[3]').text
    text=content+responding
    postdate=card.find_element_by_xpath('.//time').get_attribute('datetime')
    like_n=card.find_element_by_xpath('.//div[@data-testid="like"]').text
    retweet_n=card.find_element_by_xpath('.//div[@data-testid="retweet"]').text
    reply_n=card.find_element_by_xpath('.//div[@data-testid="reply"]').text
    tweet=(username,twitterID,text,postdate,like_n,retweet_n,reply_n)
    return tweet



#for loop
tweet_data=[]
tweet_count=set()
last_position=driver.execute_script("return window.pageYOffset;")
scrolling=True

while scrolling:
    cards=driver.find_elements_by_xpath('//article[@data-testid="tweet"]') 
    for card in cards:
#        cards=driver.find_elements_by_xpath('//article[@data-testid="tweet"]') 
        data=get_tweet_data(card)
        if data:
            tweet_index=''.join(data)
            if tweet_index not in tweet_count:
                tweet_count.add(tweet_index)
                tweet_data.append(data)
    
    scroll_attempt=0
    while True:
        #check the scroll position
        driver.execute_script('window.scrollTo(0,document.body.scrollHeight);')
        sleep(1)
        current_position=driver.execute_script("return window.pageYOffset;")
        if last_position==current_position:
            scroll_attempt +=1
            #end of scroll region
            if scroll_attempt>=10:
                scrolling=False
                break
            else:
                sleep(3)#attempt to scroll again
        else:
            last_position=current_position
            break
           
len(tweet_data)
tweet_data[-1]
dta=DataFrame(tweet_data)        
outputpath='/Users/nihz415/Desktop/final project/datacanbeused/nctmusictill701.csv'
dta.to_csv(outputpath,index=True,header=True)

with open('nct_tweets_test.csv','w',newline='',encoding='utf-8')as f:
    header=['username','twitterID','text','like_n','retweet_n,reply_n']
    writer=csv.writer(f)
    writer.writerow(header)
    writer.writerows(tweet_data)