import pandas as pd
import urllib.request
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time

mail = "andrey.vedis@gmail.com"
psw = "tw1tt3r,."

year = "2019"
month = "05"
daystart = "05"
dayend = "06"
since = year + "-" + month + "-" + daystart
until = year + "-" + month + "-" + daystart

# Code necessary to extraxt the html table we're going to scrape date from
#source_code = requests.get(url)
#soup = BeautifulSoup(plain_text, 'lxml')
#result = soup.find('div', class_ = 'Grid-cell u-size2of3 u-lg-size3of4').find('div', class_ = 'stream').find('ol', class_ = 'stream-items js-navigable-stream')
#div = result.find('div', class_ = 'stream-items js-navigable-stream" id="stream-items-id')

# With the loop below the arrays previously declared get populated
#for li in div.findAll('li', class_ = 'js-stream-item stream-item stream-item'):
#	tweet = li.find('div', class_ = 'tweet js-stream-tweet js-actionable-tweet js-profile-popup-actionable dismissible-content original-tweet js-original-tweet').find('div', class_ = 'content').find('div', class_ = 'js-tweet-text-container').find('p', class_ = 'TweetTextSize  js-tweet-text tweet-text').text
#	print(tweet)
# create a new Chrome session
driver = webdriver.Chrome()
driver.implicitly_wait(30)
driver.maximize_window()


# navigate to the application home page
driver.get("https://twitter.com/login")


# get the username textbox
login_field = driver.find_element_by_class_name("js-username-field")
login_field.clear()

# enter username
login_field.send_keys(mail)
time.sleep(1)

#get the password textbox
password_field = driver.find_element_by_class_name("js-password-field")
password_field.clear()

#enter password
password_field.send_keys(psw)
time.sleep(1)
password_field.submit()

url = 'https://twitter.com/search?q=%23long%20OR%20bull%20OR%20bullish%20OR%20moon%20OR%20skyrocket%20OR%20short%20OR%20bear%20OR%20bearish%20OR%20falling%20OR%20sinking%20%23bitcoin%20since%3A2019-05-13%20until%3A2019-05-14&src=typd&lang=en'
driver.get(url)
time.sleep(1)
body = driver.find_element_by_tag_name('body')

for _ in range(30):
    body.send_keys(Keys.PAGE_DOWN)
    time.sleep(0.2)
    tweets = driver.find_elements_by_class_name('tweet-text')

for tweet in tweets:
	# We remove hashtags because sometimes people write, i.e., '#moon', this would make our algorithm fail in counting this
	# word. By removing the '#' at the beginning, 'moon' becomes a regular word that can be counted
    dirty_text = tweet.text
    #clean_text = dirty_text.str.replace("[^a-zA-Z# ]", "")
    print(dirty_text)