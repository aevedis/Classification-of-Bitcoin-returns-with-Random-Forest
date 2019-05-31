import random
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


money = []
wealth = 100
btc_amount = wealth / test['open'].iloc[0]
print("Price of BTC at the moment of buying: " + str(test['open'].iloc[0]))
print("\n")
print("Amount of BTC bought: " + str(btc_amount))

for i in test['pricevar'].index:

	wealth = wealth + wealth * test['pricevar'][i]/100
	money.append(wealth)

#print(money)



plt.plot(money)
plt.xlabel('Dates')
plt.ylabel('Price')
plt.show()

    








