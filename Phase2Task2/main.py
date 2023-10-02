import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
#Read test_data and train_data csv files
train_data=pd.read_csv('train_data.csv')
test_data=pd.read_csv('test_data.csv')

#Drop rows which contain missing values
train_data=train_data.dropna(axis=0)
test_data=test_data.dropna(axis=0)

#Assign data as individual columns
x_train=train_data.drop('y',axis=1)
y_train=train_data['y']

x_test=test_data.drop('y',axis=1)
y_test=test_data['y']

#Creating a Linear Regression model
model=LinearRegression()
model.fit(x_train, y_train)


# Predict the output for train and test data
train_predictions = model.predict(x_train)
test_predictions = model.predict(x_test)

# Measure the performance of the model using Mean Squared error
test_mse = mean_squared_error(y_test, test_predictions)

# Measure the performance of the model using Mean Absolute error
test_mae = mean_absolute_error(y_test, test_predictions)

# Measure the performance of the model using Mean Absolute error
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

#Printing the performance metrics such as MSE,intercept, Coefficient, MAE, RMSE
print("Mean Squared Error:", test_mse)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)
print("Mean Absolute Error:", test_mae)
print("Root Mean Squared Error:", test_rmse)

#Plotting the predictions of test data using scatter plot
plt.scatter(y_test, test_predictions)
plt.title("Test data Predictions")
plt.show()

#Histogram of Residuals
plt.hist(y_test -(test_predictions))
plt.title("Histogram of residuals")
plt.show()