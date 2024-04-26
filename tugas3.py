# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Fungsi untuk membandingkan model regresi
def compare_regression_models(y_test, y_pred_LR, y_pred_DT, y_pred_RF, y_pred_SVR_linear, y_pred_SVR_poly, y_pred_SVR_rbf):
    # Mean Squared Error
    mse_LR = mean_squared_error(y_test, y_pred_LR)
    mse_DT = mean_squared_error(y_test, y_pred_DT)
    mse_RF = mean_squared_error(y_test, y_pred_RF)
    mse_SVR_linear = mean_squared_error(y_test, y_pred_SVR_linear)
    mse_SVR_poly = mean_squared_error(y_test, y_pred_SVR_poly)
    mse_SVR_rbf = mean_squared_error(y_test, y_pred_SVR_rbf)
    
    # R-squared Score
    r2_LR = r2_score(y_test, y_pred_LR)
    r2_DT = r2_score(y_test, y_pred_DT)
    r2_RF = r2_score(y_test, y_pred_RF)
    r2_SVR_linear = r2_score(y_test, y_pred_SVR_linear)
    r2_SVR_poly = r2_score(y_test, y_pred_SVR_poly)
    r2_SVR_rbf = r2_score(y_test, y_pred_SVR_rbf)
    
    # Mean Absolute Percentage Error
    mape_LR = mean_absolute_percentage_error(y_test, y_pred_LR)
    mape_DT = mean_absolute_percentage_error(y_test, y_pred_DT)
    mape_RF = mean_absolute_percentage_error(y_test, y_pred_RF)
    mape_SVR_linear = mean_absolute_percentage_error(y_test, y_pred_SVR_linear)
    mape_SVR_poly = mean_absolute_percentage_error(y_test, y_pred_SVR_poly)
    mape_SVR_rbf = mean_absolute_percentage_error(y_test, y_pred_SVR_rbf)
    
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
    
    print("\nSupport Vector Regression (Linear Kernel):")
    print("  Mean Squared Error:", mse_SVR_linear)
    print("  R-squared Score:", r2_SVR_linear)
    print("  Mean Absolute Percentage Error:", mape_SVR_linear)
    
    print("\nSupport Vector Regression (Polynomial Kernel):")
    print("  Mean Squared Error:", mse_SVR_poly)
    print("  R-squared Score:", r2_SVR_poly)
    print("  Mean Absolute Percentage Error:", mape_SVR_poly)
    
    print("\nSupport Vector Regression (RBF Kernel):")
    print("  Mean Squared Error:", mse_SVR_rbf)
    print("  R-squared Score:", r2_SVR_rbf)
    print("  Mean Absolute Percentage Error:", mape_SVR_rbf)
    
    # Menentukan model terbaik berdasarkan evaluasi
    models = {
        'Simple Linear Regression': (mse_LR, r2_LR, mape_LR),
        'Decision Tree Regression': (mse_DT, r2_DT, mape_DT),
        'Random Forest Regression': (mse_RF, r2_RF, mape_RF),
        'Support Vector Regression (Linear Kernel)': (mse_SVR_linear, r2_SVR_linear, mape_SVR_linear),
        'Support Vector Regression (Polynomial Kernel)': (mse_SVR_poly, r2_SVR_poly, mape_SVR_poly),
        'Support Vector Regression (RBF Kernel)': (mse_SVR_rbf, r2_SVR_rbf, mape_SVR_rbf)
    }
    
    best_model = min(models, key=models.get)
    return f"The best model is {best_model}."

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

# Support Vector Regression Model (Linear Kernel)
SVR_linear = SVR(kernel='linear')
SVR_linear.fit(X_train.reshape(-1, 1), y_train.ravel())
y_pred_SVR_linear = SVR_linear.predict(X_test.reshape(-1, 1))

# Support Vector Regression Model (Polynomial Kernel)
SVR_poly = SVR(kernel='poly')
SVR_poly.fit(X_train.reshape(-1, 1), y_train.ravel())
y_pred_SVR_poly = SVR_poly.predict(X_test.reshape(-1, 1))

# Support Vector Regression Model (RBF Kernel)
SVR_rbf = SVR(kernel='rbf')
SVR_rbf.fit(X_train.reshape(-1, 1), y_train.ravel())
y_pred_SVR_rbf = SVR_rbf.predict(X_test.reshape(-1, 1))

# Panggil fungsi untuk membandingkan model
result = compare_regression_models(y_test, y_pred_LR, y_pred_DT, y_pred_RF, y_pred_SVR_linear, y_pred_SVR_poly, y_pred_SVR_rbf)
print(result)
