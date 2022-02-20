# Simple-linear-regression-
Predict the scores of students based on the no. of study hours
Created on Sat Feb 12 23:59:41 2022

@author: admin pc
"""


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline

# Reading data from remote link
url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)
data.describe()
data.info()

# Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# preparing data 
# diving data into attributes(inputs) and labels(outputs)

X = data.iloc[:, :-1].values
y = data.iloc[:, 1].values
 
#now splitting the data into training and test sets

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=0) 

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

# Training the algorithm

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 



print("Training complete.")

# Plotting the regression line

line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color = "purple");
plt.show()

## Prediction
# we have trained our data now its time to make some prdictions

print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores
print(y_pred)

from pandas import(DataFrame)

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df
df1=pd.DataFrame(df)

#Difference between predicted values and actual values

difference = df['Actual']-df['Predicted']
print(difference)

# You can also test with your own data
# task to predict the score of the student who studies for 9.25 hours

print("score of a student who stuides for 9.25 hours per day is", regressor.predict([[9.25]]))




from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)) 
