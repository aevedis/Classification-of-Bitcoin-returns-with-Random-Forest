import os
import pandas as pd
import urllib.request
from collections import Counter
from numpy import random
import re
import string
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import datetime

bull_vocab = ["long", "bull", "bullish", "moon", "skyrocket"]
bear_vocab = ["short", "bear", "bearish", "falling", "sinking"]
since = "2016-01-01"
until = "2016-01-02"
numbullwords = 0
numbearwords = 0

# Function useful to remove undesired punctuation of the form: "!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~",
# multiple spaces, trailing newlines and spaces at the beginning of the string. Every word is also forced to lowercase 
def clean_text(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = text.replace("\n", " ")
    text = re.sub('[0-9]+', ' ', text)
    text = re.sub(' +',' ', text)
    text = text.lstrip()
    text = text.lower()
    return text

def generate_dates(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    df = pd.DataFrame({'Date': date_generated})
    return df


# create a new Chrome session
driver = webdriver.Chrome()
driver.implicitly_wait(30)
#driver.maximize_window()

# Generating array of dates
dates_df = generate_dates(since, "2017-01-01") # IMPORTANT: the second argument is the final that we wish to loop
for i in dates_df.index:
    if i>0:
        since = dates_df['Date'][i-1].strftime("%Y-%m-%d")
        until = dates_df['Date'][i].strftime("%Y-%m-%d")

    # navigate to the application home page
    driver.get("https://twitter.com")

    url = 'https://twitter.com/search?q=%23long%20OR%20bull%20OR%20bullish%20OR%20moon%20OR%20skyrocket%20OR%20short%20OR%20bear%20OR%20bearish%20OR%20falling%20OR%20sinking%20%23bitcoin%20since%3A' + since + '%20until%3A' + until + '&src=typd&lang=en'
    driver.get(url)
    sleeptime = random.uniform(1.0, 3.0)
    time.sleep(sleeptime)
    body = driver.find_element_by_tag_name('body')

    for _ in range(40):
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.2)
        tweets = driver.find_elements_by_class_name('tweet-text')

        tweets_df = pd.DataFrame({'Tweet': [tweet.text for tweet in tweets]})

        tweets_df['Tweet'] = tweets_df['Tweet'].apply(lambda x: clean_text(x))


    for i in tweets_df.index:
        wordcountbull = dict((x,0) for x in bull_vocab)
        wordcountbear = dict((x,0) for x in bear_vocab)
        for w in re.findall(r"\w+", str(tweets_df['Tweet'][i])):
            if w in wordcountbull:
                numbullwords += 1
            elif w in wordcountbear:
                numbearwords += 1

    # We put all the data in Pandas' DataFrame object. Mathematically, a simple matrix
    df = pd.DataFrame({'Date': [since],'NumBullWords': [numbullwords], 'NumBearWords': [numbearwords]})
    print("Date: " + str(since) + "\n" + "NumBullWords: " + str(numbullwords) + "\n" + "NumBearWords: " + str(numbearwords) + "\n" + "\n")
    
    # Setting back the counters to zero for the new cycle
    numbullwords = 0
    numbearwords = 0

    # Ff file doesn't exist, write header 
    if not os.path.isfile('tweet_data.csv'):
       df.to_csv('tweet_data.csv', index=False)
    else: # If it exists, append new data without writing header
       df.to_csv('tweet_data.csv', mode='a', header=False, index=False)

