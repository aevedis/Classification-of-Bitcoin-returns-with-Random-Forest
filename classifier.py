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
labels = df.columns[:25]

# Assigning class' values to y variable
y = train['class']
X = df.drop('class', 1)

my_X, my_y = X, df['class']

start = time.time()

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators = 600, random_state = 73)

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
	
print(yprob)
print("\n")
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("\n")
print(pd.crosstab(y_actual, y_predicted, rownames=['Actual classes'], colnames=['Predicted classes']))


# Plot Learning curve
#skplt.estimators.plot_learning_curve(clf, my_X, my_y, cv=5)
#plt.show()

# Plot ROC curve
skplt.metrics.plot_roc(y_actual, yprob)
plt.show()

# Plot Precision Recall Curve
#skplt.metrics.plot_precision_recall_curve(y_actual, yprob)
#plt.show()

# Plot confusion matrix
skplt.metrics.plot_confusion_matrix(y_true=y_actual, y_pred=y_predicted)
plt.show()
