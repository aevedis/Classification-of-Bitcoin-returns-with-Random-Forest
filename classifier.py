import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np
import time

# Read in data and display first 5 rows
dataset = pd.read_csv('Data/dataset.csv')
df = pd.DataFrame(dataset)
#df = pd.to_numeric(df)
#df = df.astype(np.float16)
#df=df.apply(pd.to_numeric, errors='coerce')


# Removing the first column (Date always increments, hence it is not useful for predicting)
df = df.drop('date', 1)

# Printing descriptive statistics about the dataset
#print(df.describe())


train, test = train_test_split(df, test_size=0.2)

# Creating a list of the feature column's names
labels = df.columns[:16]

# Assigning class' values to y variable
y = train['class']

start = time.time()

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators = 600, random_state = 73)

# Training the classifier on our test dataset
clf.fit(train[labels], y)

end = time.time()
executiontime = end - start
executiontimestr = str(executiontime)
print("Random Forest successfully trained. The process took " + executiontimestr + " seconds")

yactual = test['class']
ypredicted = clf.predict(test[labels])

	
#print(ypredicted)
print(clf.predict_proba(test[labels])[0:50])
print("\n")
print(pd.crosstab(yactual, ypredicted, rownames=['Actual classes'], colnames=['Predicted classes']))

skplt.metrics.plot_confusion_matrix(yactual, ypredicted, normalize=True)
plt.show()

# Plot AUC Curve: how, considering that they are made for binary classification?
#y_pred_proba = clf.predict_proba(test[labels])[::,1]
#fpr, tpr, _ = metrics.roc_curve(yactual,  ypredicted)
#auc = metrics.roc_auc_score(yactual, ypredicted)
#plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#plt.legend(loc=4)
#plt.show()
