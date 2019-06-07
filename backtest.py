import random
from random import randint
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
test_size = 0.2
dataset = pd.read_csv('Data/dataset.csv')
df = pd.DataFrame(dataset)


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









# Creating a list of the feature columns' names
labels = df.columns[:25]

# Assigning class' values to y variable
y = train['class']
X = df.drop('class', 1)

my_X, my_y = X, df['class']

start = time.time()

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators = 600, max_depth = 180, max_features=15, random_state = 73)

# Training the classifier on our test dataset
clf.fit(train[labels], y)

end = time.time()
executiontime = end - start
executiontimestr = str(executiontime)
print("Random Forest successfully trained. The process took " + executiontimestr + " seconds")

y_actual = test['class']
y_predicted = clf.predict(test[labels])
predictions = [round(value) for value in y_predicted]
n_classes = 4
yprob = clf.predict_proba(test[labels])
accuracy = accuracy_score(y_actual, predictions)
	
#print(yprob)
print("\n")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("\n")
print(pd.crosstab(y_actual, y_predicted, rownames=['Actual classes'], colnames=['Predicted classes']))

# Plot ROC curve
skplt.metrics.plot_roc(y_actual, yprob)
plt.show()

# Plot Learning curve
#skplt.estimators.plot_learning_curve(clf, my_X, my_y, cv=5)
#plt.show()

# Plot Precision Recall Curve
#skplt.metrics.plot_precision_recall_curve(y_actual, yprob)
#plt.show()

# Plot confusion matrix
skplt.metrics.plot_confusion_matrix(y_true=y_actual, y_pred=y_predicted)
plt.show()



# Getting actual price for each day
BH_portfolio_values = []
actual_portfolio_values = []
dates = []
prices = []

# The two SLOC below are necessary because of the construction of the dataset
num = num +1
num2 = num2 +1

dataset = pd.read_csv('Data/bitcoin_price_data.csv')
df2 = pd.DataFrame(dataset) # size 1970
USD_holdings = 1000
BTC_holdings = 0
benchmark_BTC_holdings = USD_holdings / df2['open'][num]
wealth = 0
for row in range(len(df2)):
    if row >= num and row <= num2:
        date = df2['date'][row]
        dates.append(date)

        price = df2['open'][row]
        prices.append(price)


       
    else:
    	pass


df3 = pd.DataFrame({'date': dates,
'price': prices,
})

df3['prediction'] = y_predicted
print(df3)


for row in range(len(df3)):

    if df3['prediction'][row] == 1:
        #Short 2x
        if BTC_holdings > 0:
            USD_holdings = BTC_holdings * df3['price'][row]
            BTC_holdings = 0
    elif df3['prediction'][row] == 2:
    	#Short 1.5x
        if BTC_holdings > 0:
            USD_holdings = BTC_holdings * df3['price'][row]
            BTC_holdings = 0
    elif df3['prediction'][row] == 3:
    	#Sell no lev
        if BTC_holdings > 0:
            USD_holdings = BTC_holdings * df3['price'][row]
            BTC_holdings = 0
    elif df3['prediction'][row] == 4:
    	#Buy no lev
        if USD_holdings > 0:
            BTC_holdings = USD_holdings / df3['price'][row]
            USD_holdings = 0
    elif df3['prediction'][row] == 5:
    	#Long 1.5x
        if USD_holdings > 0:
            BTC_holdings = USD_holdings / df3['price'][row]
            USD_holdings = 0
    elif df3['prediction'][row] == 6:
    	#Long 2x
        if USD_holdings > 0:
            BTC_holdings = USD_holdings / df3['price'][row]
            USD_holdings = 0

    wealth = BTC_holdings * df3['price'][row] + USD_holdings
    actual_portfolio_values.append(wealth)

    BH_wealth = benchmark_BTC_holdings * df3['price'][row] 
    BH_portfolio_values.append(BH_wealth)

#plot([go.Scatter(x=dates, y=portfolio_values, marker={'color': 'blue'})])

#iplot([{"x": dates, "y": portfolio_values}]) # For Jupyter Notebook

#iplot(cf.datagen.lines().iplot(asFigure=True, Plot with cufflinks
#                               kind='scatter',xTitle='Dates',yTitle='Returns',title='Returns'))

print(actual_portfolio_values)


# Create a trace
tracebench = go.Scatter(
	x = dates,
    y = BH_portfolio_values,
	mode = 'lines',
	name = 'Benchmark',
    line = dict(
        color = ('rgb(180, 46, 71)'),
        width = 4)
)

traceportfolio = go.Scatter(
    x = dates,
    y = actual_portfolio_values,
    mode = 'lines',
    name = 'Portfolio',
    line = dict(
        color = ('rgb(45, 78, 175)'),
        width = 4)
)

data = [tracebench, traceportfolio]

layout = dict(title = 'Bitcoin wealth variation in absolute prices',
              xaxis = dict(title = 'Date'),
              yaxis = dict(title = 'Wealth expressed in USD'),
              )

fig = dict(data=data, layout=layout)

py.iplot(fig, filename='sysinv-bitcoin')