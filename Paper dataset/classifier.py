import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import scikitplot as skplt
import pandas as pd
import numpy as np
import time

test_size = 0.2
dataset = pd.read_csv('finaldataset.csv')
df = pd.DataFrame(dataset)
df = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
dates = []

# Splitting dataset into train and test sets
train, test = train_test_split(df, test_size=0.2)

# Creating a list of the feature columns' names
labels = df.columns[:116]

# Assigning class' values to y variable
y = train['y']
X = df.drop('y', 1)

my_X, my_y = X, df['y']

start = time.time()

# Create a random forest Classifier. By convention, clf means 'Classifier'
clf = RandomForestClassifier(n_estimators = 600, max_depth = 180, max_features=15, random_state = 73)

# Training the classifier on our test dataset
clf.fit(train[labels], y)

end = time.time()
executiontime = end - start
executiontimestr = str(executiontime)
print("Random Forest successfully trained. The process took " + executiontimestr + " seconds")

y_actual = test['y']
y_pred_clf = clf.predict(test[labels])
predictions_clf = [round(value) for value in y_pred_clf]
n_classes = 4
yprob = clf.predict_proba(test[labels])
accuracy_clf = accuracy_score(y_actual, predictions_clf)
	

#print(yprob)
print("\n")
print("Accuracy CLF: %.2f%%" % (accuracy_clf * 100.0))
print("\n")
print(pd.crosstab(y_actual, y_pred_clf, rownames=['Actual classes'], colnames=['Predicted classes']))



#Plot ROC curve
skplt.metrics.plot_roc(y_actual, yprob)
plt.show()

# Plot Learning curve
#skplt.estimators.plot_learning_curve(clf, my_X, my_y, cv=5)
#plt.show()

# Plot Precision Recall Curve
#skplt.metrics.plot_precision_recall_curve(y_actual, yprob)
#plt.show()

# Plot confusion matrix
skplt.metrics.plot_confusion_matrix(y_true=y_actual, y_pred=y_pred_clf)
plt.show()

# We put all the data in Pandas' DataFrame object. Mathematically, a simple matrix
df = pd.DataFrame({
'prediction': y_pred_clf,
})

# Data is put into a csv file
df.to_csv('predictions.csv', index=False)


# Use the previous 10 bars' movements to predict the next movement.
 
# Use a random forest classifier. More here: http://scikit-learn.org/stable/user_guide.html
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import numpy as np
 
def initialize(context):
    context.security = sid(698) # Boeing
    context.window_length = 10 # Amount of prior bars to study
    
    context.classifier = RandomForestClassifier() # Use a random forest classifier
    
    # deques are lists with a maximum length where old entries are shifted out
    context.recent_prices = deque(maxlen=context.window_length+2) # Stores recent prices
    context.X = deque(maxlen=500) # Independent, or input variables
    context.Y = deque(maxlen=500) # Dependent, or output variable
    
    context.prediction = 0 # Stores most recent prediction
    
def handle_data(context, data):
    context.recent_prices.append(data[context.security].price) # Update the recent prices
    if len(context.recent_prices) == context.window_length+2: # If there's enough recent price data
        
        # Make a list of 1's and 0's, 1 when the price increased from the prior bar
        changes = np.diff(context.recent_prices) > 0
        
        context.X.append(changes[:-1]) # Add independent variables, the prior changes
        context.Y.append(changes[-1]) # Add dependent variable, the final change
        
        if len(context.Y) >= 100: # There needs to be enough data points to make a good model
            
            context.classifier.fit(context.X, context.Y) # Generate the model
            
            context.prediction = context.classifier.predict(changes[1:]) # Predict
            
            # If prediction = 1, buy all shares affordable, if 0 sell all shares
            order_target_percent(context.security, context.prediction)
                
            record(prediction=int(context.prediction))