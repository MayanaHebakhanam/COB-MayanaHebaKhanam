import pandas as pd

#Read the csv file and store in variable dataframe
dataframe=pd.read_csv("dataset - netflix1.csv")

#Replace {'Not Given', ' ', '?'} with NA
dataframe=dataframe.replace('Not Given', "NA")
dataframe=dataframe.replace(' ', "NA")
dataframe=dataframe.replace('?', "NA")

#Use interquantile to remove outliers
q1=dataframe.quantile(0.25)
q2=dataframe.quantile(0.95)
IQR=q2-q1
dataframe=dataframe[~((dataframe < (q1 - 1.5 *IQR)) | (dataframe > (q2 + 1.5 * IQR))).any(axis=1)]

#Now store the changed data into a new file say "new-dataset.csv"
dataframe.to_csv('new-dataset.csv', index=False)