import pandas as pd
import tweepy as tw
import urllib.request
import re
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time

#consumer_key= 'cWUc4ExAjp6yBw2wzwtdmLruM'
#consumer_secret= '4xcGdhUSnnOAdYK5wVhJbUHrt1dsEwRyeiWrVLDzQU1eOrq8na'
#access_token= '3026636429-TS71HXTXtnjTUGEf9ee45DQe3h5SXx4KHD9sK4O'
#access_token_secret= 'MxOzLX2HMYtq65efkjsapiIdFB3SMJHqJO7gomUXzx2OF'

#url = 'https://twitter.com/search?l=&q=moon%20OR%20mooning%20OR%20balloon%20OR%20ballooning%20OR%20skyrocket%20OR%20skyrocketing%20OR%20hodl%20OR%20hold%20OR%20buy%20OR%20bullish%20OR%20bull%20-sell%20-bear%20-bearish%20%23bitcoin%20OR%20%23btc%20since%3A2019-05-05%20until%3A2019-05-06&src=typd&lang=en'

url = 'https://twitter.com/search?l=&q=moon%20OR%20mooning%20OR%20balloon%20OR%20ballooning%20OR%20skyrocket%20OR%20skyrocketing%20OR%20hodl%20OR%20hold%20OR%20buy%20OR%20bullish%20OR%20bull%20-sell%20-bear%20-bearish%20%23bitcoin%20OR%20%23btc%20since%3A2019-05-05%20until%3A2019-05-06&src=typd&lang=en'
#website_url = requests.get(url).text
    

#soup = BeautifulSoup(website_url,'lxml')
#print(soup.prettify())

driver = webdriver.Firefox()
driver.get("https://www.python.org/")
driver.execute_script('document.title')