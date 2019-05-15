import pandas as pd
import time
import re
import csv
import requests
from bs4 import BeautifulSoup

# I arbitrarily decided to scrape data from January 1st 2014
start_date = "20140101"

# Sets the variable "today" as the datetime object.
today = time.strftime('%Y%m%d')

# The url is dynamically generated based on current date, and the start date chosen at codeline 9
url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + start_date + "&end=" + today

# Code necessary to extraxt the html table we're going to scrape date from
source_code = requests.get(url)
plain_text = source_code.text
soup = BeautifulSoup(plain_text, 'html.parser')
table_body = soup.find('tbody')

dates = []
open_prices = []
close_prices = []
volumes = []

# With the loop below the arrays previously declared get populated
for row in table_body.findAll('tr'):
        td = row.findAll('td')
        date = td[0].text
        dates.append(date)
        open_price = td[1].text
        open_prices.append(open_price)
        close_price = td[4].text
        close_prices.append(close_price)
        volume = td[5].text
        volumes.append(volume)


        
# We put all the data in Pandas' DataFrame object. Mathematically, a simple matrix
df = pd.DataFrame({'date': dates,
'open': open_prices,
'close': close_prices,
'volume': volumes,
})

# Data is put into a csv file
df.to_csv('bitcoin_price_data.csv')
