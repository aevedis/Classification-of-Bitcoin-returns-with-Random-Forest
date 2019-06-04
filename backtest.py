import random
import csv
import datetime
import cufflinks as cf
import plotly
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.plotly as py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np
import seaborn as sns; sns.set()
import time

# Authenticate with Plotly
plotly.tools.set_credentials_file(username='AndreyVedis94',                                              
                                  api_key='4JII7NXfEX71SWNuRGrO')

# Read in data and display first 5 rows
dataset = pd.read_csv('Data/dataset.csv')
df = pd.DataFrame(dataset)
test_size = 0.2


# Removing the first column (Date always increments, hence it is not useful for predicting)
df = df.drop('date', 1)

# We generate the cutoff at which we extract a sample of the dataset that will be used to test our
# trading strategy on
num = random.randint(0,len(df)*(1-test_size))
num2 = num + len(df) * test_size
print(str(num) + "\n" + str(num2) + "\n")

is_test = []

for i in range(len(df)):
    is_test.append(1) if i >= num and i <= num2 else is_test.append(0)

df['is_test'] = is_test
    
# Create two new dataframes, one with the training rows, one with the test rows
train, test = df[df['is_test']==0], df[df['is_test']==1]

# Removing 'is_test' column considering that it's not useful anymore
train = train.drop('is_test', 1)
test = test.drop('is_test', 1)

# Show the number of observations for the test and training dataframes
print('Number of observations in the training data:', len(train))
print('Number of observations in the test data:',len(test))
print("\n")


# Getting actual price for each day
portfolio_values = []
dates = []

# The two SLOC below are necessary because of the construction of the dataset
num = num +1
num2 = num2 +1

dataset = pd.read_csv('Data/bitcoin_price_data.csv')
df2 = pd.DataFrame(dataset)
usd_holdings = 1000
btc_holdings = usd_holdings / df2['open'][num]
usd_holdings = 0
portfolio_value = 0
wealth = 0
for row in range(len(df2)):
    if row >= num and row <= num2:
        date = df2['date'][row]
        dates.append(date)
        wealth = btc_holdings * df2['open'][row]
        portfolio_values.append(wealth)
    else:
    	pass


print(df['open'])
#plot([go.Scatter(x=dates, y=portfolio_values, marker={'color': 'blue'})])

#iplot([{"x": dates, "y": portfolio_values}]) # For Jupyter Notebook

#iplot(cf.datagen.lines().iplot(asFigure=True, Plot with cufflinks
#                               kind='scatter',xTitle='Dates',yTitle='Returns',title='Returns'))


# Create a trace
trace = go.Scatter(
	name = 'Benchmark',
    x = dates,
    y = portfolio_values,
    line = dict(
        color = ('rgb(180, 46, 71)'),
        width = 4,
        dash = 'dot')
)

data = [trace]

layout = dict(title = 'Bitcoin wealth variation in absolute prices',
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Wealth expressed in USD'),
              )

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='sysinv-bitcoin')