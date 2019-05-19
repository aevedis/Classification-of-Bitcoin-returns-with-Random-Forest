from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import time

# Setting random seed
np.random.seed(73)

# Loading the dataset into a Pandas' DataFrame object
dataset = pd.read_csv('dataset.csv')
df = pd.DataFrame(dataset)

# Some descriptive statistics about every column
#print(df.describe())

# Removing the first column (Date always increments, hence it is not useful for predicting)
df = df.drop('date', 1)

# Creating two separate datasets, the first one to train our random forest, the second one to test it on unseen data
train, test = df[df['istest']==0], df[df['istest']==1]

#print('Number of observations in the training data:', len(train))
#print('Number of observations in the test data:',len(test))

# Creating a list of the feature column's names
labels = df.columns[:7]


# Assigning class' values to y variable
y = train['class']

start = time.time()

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=50)

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

#print(list(zip(train[labels], clf.feature_importances_)))

#print(pd.crosstab(yactual, ypredicted, rownames=['Actual classes'], colnames=['Predicted classes']))