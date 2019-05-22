import os
import pandas as pd
import urllib.request
from collections import Counter
import re
import string
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

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

year = "2019"
month = "05"
daystart = "05"
dayend = "06"
since = year + "-" + month + "-" + daystart
until = year + "-" + month + "-" + daystart
bull_vocab = ["long", "bull", "bullish", "moon", "skyrocket"]
bear_vocab = ["short", "bear", "bearish", "falling", "sinking"]
numbullwords = 0
numbearwords = 0


# With the loop below the arrays previously declared get populated
#for li in div.findAll('li', class_ = 'js-stream-item stream-item stream-item'):
#	tweet = li.find('div', class_ = 'tweet js-stream-tweet js-actionable-tweet js-profile-popup-actionable dismissible-content original-tweet js-original-tweet').find('div', class_ = 'content').find('div', class_ = 'js-tweet-text-container').find('p', class_ = 'TweetTextSize  js-tweet-text tweet-text').text
#	print(tweet)
# create a new Chrome session
driver = webdriver.Chrome()
driver.implicitly_wait(30)
driver.maximize_window()


# navigate to the application home page
driver.get("https://twitter.com")

url = 'https://twitter.com/search?q=%23long%20OR%20bull%20OR%20bullish%20OR%20moon%20OR%20skyrocket%20OR%20short%20OR%20bear%20OR%20bearish%20OR%20falling%20OR%20sinking%20%23bitcoin%20since%3A2019-05-09%20until%3A2019-05-10&src=typd&lang=en'
driver.get(url)
time.sleep(1)
body = driver.find_element_by_tag_name('body')

for _ in range(15):
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
df = pd.DataFrame({'NumBullWords': [numbullwords], 'NumBearWords': [numbearwords]})
print("NumBullWords: " + str(numbullwords) + "\n" + "NumBearWords: " + str(numbearwords) + "\n" + "\n")

# Ff file doesn't exist, write header 
if not os.path.isfile('tweet_data.csv'):
   df.to_csv('tweet_data.csv', index=False)
else: # If it exists, append new data without writing header
   df.to_csv('tweet_data.csv', mode='a', header=False, index=False)

