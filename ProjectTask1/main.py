import requests
import pandas as pd

#Get the url of the data
url="https://data.binance.com/api/v3/ticker/24hr"
response=requests.get(url)

#Store the response in the form of JSON
data=response.json()

#Convert the data in to DataFrame
dataframe=pd.DataFrame(data)

#Convert the above obtained DataFrame into CSV and store into a csv file "binance-data.csv"
dataframe.to_csv("binance-data.csv", index=False)

