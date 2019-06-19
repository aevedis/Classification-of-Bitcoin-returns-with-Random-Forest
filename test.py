import random
from random import randint
import csv
import datetime
import pandas as pd
import numpy as np
import time

# Getting actual price for each day
BH_portfolio_values = []
actual_portfolio_values = []
dates = []
coin_array = []
prices = []
cash_array = []
levcash_array = []
signals = [6, 4, 5, 4, 4, 3, 2, 1, 2, 6, 5, 1, 3, 2, 4, 5, 2, 6, 6, 5]

'''for i in range(20):
	signals.append(randint(1, 6))
'''
dataset = pd.read_csv('Data/bitcoin_price_data.csv')
df2 = pd.DataFrame(dataset)
USD_holdings = 1000
BTC_holdings = 0
benchmark_BTC_holdings = USD_holdings / df2['open'][0]
wealth = 0
margin = 0

for row in range(20):

    date = df2['date'][row]
    dates.append(date)

    

    if signals[row] == 1:
        #Short 2x
        if BTC_holdings > 0:
            USD_holdings = BTC_holdings * df2['open'][row] - margin
            BTC_holdings = 0
            margin = 0
    elif signals[row] == 2:
        #Short 1.5x
        if BTC_holdings > 0:
            USD_holdings = BTC_holdings * df2['open'][row] - margin
            BTC_holdings = 0
            margin = 0
    elif signals[row] == 3:
        #Sell no lev
        if BTC_holdings > 0:
            USD_holdings = BTC_holdings * df2['open'][row] - margin
            BTC_holdings = 0
            margin = 0
    elif signals[row] == 4:
        #Buy no lev
        if USD_holdings > 0:
            BTC_holdings = USD_holdings / df2['open'][row]
            USD_holdings = 0
    elif signals[row] == 5:
        #Long 1.5x
        if USD_holdings > 0:
            margin = USD_holdings * 0.5
            BTC_holdings = (USD_holdings + margin) / df2['open'][row]
            USD_holdings = 0
    elif signals[row] == 6:
        #Long 2x
        if USD_holdings > 0:
            margin = USD_holdings
            BTC_holdings = (USD_holdings + margin) / df2['open'][row]
            USD_holdings = 0
    
    wealth = BTC_holdings * df2['open'][row] + USD_holdings - margin
    actual_portfolio_values.append(wealth)

    BH_wealth = benchmark_BTC_holdings * df2['open'][row] 
    BH_portfolio_values.append(BH_wealth)
    levcash_array.append(margin)
    prices.append(df2['open'][row])
    coin_array.append(BTC_holdings)
    cash_array.append(USD_holdings)

# We put all the data in Pandas' DataFrame object. Mathematically, a simple matrix
df = pd.DataFrame({'date': dates,
'benchmark': BH_portfolio_values,
'open': prices,
'BTC': coin_array,
'cash': cash_array,
'margin': levcash_array,
'wealth': actual_portfolio_values,
'signal': signals,
})
    
print(df)