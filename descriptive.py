import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Read in data and display first 5 rows
dataset = pd.read_csv('Data/dataset.csv')
df = pd.DataFrame(dataset)
#df = pd.to_numeric(df)
#df = df.astype(np.float16)
#df=df.apply(pd.to_numeric, errors='coerce')


# matplotlib histogram
plt.hist(df['class'], color = 'red', edgecolor = 'black',
         bins = int(180/15))


# seaborn histogram
sns.distplot(df['class'], hist=True, kde=False, 
             bins=int(180/15), color = 'blue',
             hist_kws={'edgecolor':'black'})
# Add labels
plt.title('Class distribution')
plt.xlabel('Class')
plt.ylabel('Occurrences')
plt.show()
