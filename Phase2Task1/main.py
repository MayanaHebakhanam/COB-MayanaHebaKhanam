import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

dataf=pd.read_csv("dataset - netflix1.csv")

#Analyze the dataset
#Step-1: Displaying first few rows
print(dataf.head())

#Step-2: Get summary of the dataset
print(dataf.describe())

#Step-3: Check for missing values
print(dataf.isnull().sum())

#Replace {'Not Given', ' ', '?'} with NA
dataf=dataf.replace('Not Given', np.NaN)
dataf=dataf.replace(' ', np.NaN)
dataf=dataf.replace('?', np.NaN)

#Use interquantile to remove outliers
q1=dataf.quantile(0.25)
q2=dataf.quantile(0.95)
IQR=q2-q1
dataf=dataf[~((dataf < (q1 - 1.5 *IQR)) | (dataf > (q2 + 1.5 * IQR))).any(axis=1)]

#Generate cleaned dataset
dataf.to_csv('new-dataset.csv', index=False)
dataf=pd.read_csv("new-dataset.csv")

# 1. Scatter plot
sb.scatterplot(x=dataf['rating'], y=dataf['release_year'])
plt.title('Scatter Plot')
plt.show()

# 2. Histogram
sb.histplot(data=dataf, x='release_year', kde=True)
plt.title('Histogram')
plt.show()

# 3. Box plot
sb.boxplot(x=dataf['rating'], y=dataf['release_year'])
plt.title('Box Plot')
plt.show()

# 4. Bar plot
sb.barplot(x=dataf['rating'], y=dataf['release_year'],palette='Blues')
dataf.groupby(['rating']).mean()
plt.title('Bar Plot')
plt.show()

# 5. Count plot
sb.countplot(x=dataf['rating'])
plt.title('Count Plot')
plt.show()
