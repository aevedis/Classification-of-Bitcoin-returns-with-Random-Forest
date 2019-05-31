import pandas as pd
import time
import math
import re
import csv
import requests
from bs4 import BeautifulSoup

# I arbitrarily decided to scrape data from January 1st 2014
start_date = "20131228"

# Sets the variable "today" as the datetime object.
today = time.strftime('%Y%m%d')

# The url is dynamically generated based on current date, and the start date chosen at codeline 9
url = "https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=" + start_date + "&end=" + today
start = time.time()

# Code necessary to extract the html table we're going to scrape date from
source_code = requests.get(url)
plain_text = source_code.text
soup = BeautifulSoup(plain_text, 'html.parser')
table_body = soup.find('tbody')

dates = []
open_prices = []
close_prices = []
volumes = []
variations = []
classes = []

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
        price_var = ((float(close_price)/float(open_price))-1)*100
        variations.append(price_var)
        if price_var>3.5:
            varclass=6
        elif price_var<=3.5 and price_var>1.5:
            varclass=5
        elif price_var>=0 and price_var<=1.5:
            varclass=4
        elif price_var<0 and price_var>=-1.5:
            varclass=3
        elif price_var>=-3.5 and price_var<-1.5:
            varclass=2
        elif price_var<-3.5:
        	varclass=1
        classes.append(varclass)


        
# We put all the data in Pandas' DataFrame object. Mathematically, a simple matrix
df = pd.DataFrame({'date': dates,
'open': open_prices,
'close': close_prices,
'volume': volumes,
'pricevar': variations,
'classes': classes,
})

# Data is put into a csv file
df.to_csv('bitcoin_price_data.csv', index=False)

end = time.time()
executiontime = end - start
executiontimestr = str(executiontime)
print("CSV successfully generated. The process took " + executiontimestr + " seconds")