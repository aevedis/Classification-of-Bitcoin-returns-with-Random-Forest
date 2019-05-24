import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
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
n_classes = 4

	
#print(ypredicted)
print(clf.predict_proba(test[labels])[0:50])
print("\n")
print(pd.crosstab(yactual, ypredicted, rownames=['Actual classes'], colnames=['Predicted classes']))

#skplt.metrics.plot_confusion_matrix(yactual, ypredicted, normalize=True)
#plt.show()

# 1st error
# Plot AUC Curve: how, considering that they are made for binary classification?
#y_pred_proba = clf.predict_proba(test[labels])[::,1]
#fpr, tpr, _ = metrics.roc_curve(yactual,  ypredicted)
#auc = metrics.roc_auc_score(yactual, ypredicted)
#plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#plt.legend(loc=4)
#plt.show()



# 2nd error
#skplt.metrics.plot_roc(yactual, ypredicted)
#plt.show()




# 3rd error
# Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(yactual[:, i], ypredicted[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
#for i in range(n_classes):
#    plt.figure()
#    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
#    plt.legend(loc="lower right")
#    plt.show()