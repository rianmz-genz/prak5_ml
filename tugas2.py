# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Fungsi untuk membandingkan model regresi
def compare_regression_models(y_test, y_pred_LR, y_pred_DT, y_pred_RF):
    # Mean Squared Error
    mse_LR = mean_squared_error(y_test, y_pred_LR)
    mse_DT = mean_squared_error(y_test, y_pred_DT)
    mse_RF = mean_squared_error(y_test, y_pred_RF)
    
    # R-squared Score
    r2_LR = r2_score(y_test, y_pred_LR)
    r2_DT = r2_score(y_test, y_pred_DT)
    r2_RF = r2_score(y_test, y_pred_RF)
    
    # Mean Absolute Percentage Error
    mape_LR = mean_absolute_percentage_error(y_test, y_pred_LR)
    mape_DT = mean_absolute_percentage_error(y_test, y_pred_DT)
    mape_RF = mean_absolute_percentage_error(y_test, y_pred_RF)
    
    print("Simple Linear Regression:")
    print("  Mean Squared Error:", mse_LR)
    print("  R-squared Score:", r2_LR)
    print("  Mean Absolute Percentage Error:", mape_LR)
    
    print("\nDecision Tree Regression:")
    print("  Mean Squared Error:", mse_DT)
    print("  R-squared Score:", r2_DT)
    print("  Mean Absolute Percentage Error:", mape_DT)
    
    print("\nRandom Forest Regression:")
    print("  Mean Squared Error:", mse_RF)
    print("  R-squared Score:", r2_RF)
    print("  Mean Absolute Percentage Error:", mape_RF)
    
    # Menentukan model terbaik berdasarkan evaluasi
    if mse_LR < mse_DT and mse_LR < mse_RF and r2_LR > r2_DT and r2_LR > r2_RF and mape_LR < mape_DT and mape_LR < mape_RF:
        return "Simple Linear Regression is the best model."
    elif mse_DT < mse_RF and r2_DT > r2_RF and mape_DT < mape_RF:
        return "Decision Tree Regression is the best model."
    else:
        return "Random Forest Regression is the best model."

# Import dataset
dataset = pd.read_csv('IceCreamData.csv')
X = dataset['Temperature'].values
y = dataset['Revenue'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Simple Linear Regression Model
LRregressor = LinearRegression()
LRregressor.fit(X_train.reshape(-1, 1), y_train.ravel())
y_pred_LR = LRregressor.predict(X_test.reshape(-1, 1))

# Decision Tree Regressor Model
DTregressor = DecisionTreeRegressor()
DTregressor.fit(X_train.reshape(-1, 1), y_train.ravel())
y_pred_DT = DTregressor.predict(X_test.reshape(-1, 1))

# Random Forest Regression Model
RFregressor = RandomForestRegressor(n_estimators=10, random_state=0)
RFregressor.fit(X_train.reshape(-1, 1), y_train.ravel())
y_pred_RF = RFregressor.predict(X_test.reshape(-1, 1))

# Panggil fungsi untuk membandingkan model
result = compare_regression_models(y_test, y_pred_LR, y_pred_DT, y_pred_RF)
print(result)
