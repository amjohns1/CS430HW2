##########################################################
# Class: CS 430/530
# Assignment: Homework 2
# Team Number: 10
# Team Members: Adam Barr, Adam Johnson, Scott Petty
# Date: 10/1/2023
##########################################################

# importing the required module 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# clean up data file
def clean_data_file():
    start_line = 22
    joined_lines = []
    with open('boston.txt', 'r') as file:
        for line_number, line in enumerate(file, 1):
            if line_number >= start_line:
                lines = file.readlines()

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                joined_line = lines[i].strip() + lines[i + 1]
                joined_lines.append(joined_line)

    with open('boston_cleaned.csv', 'w') as file:
        file.writelines(joined_lines)

def mean_squared_error(y_actual, y_predicted):
    # Calculating the loss or cost
    cost = (1/(2*len(y_actual))) * np.sum((y_actual - y_predicted)**2)
    return cost

def training_gradient_descent(X, y, learning_rate, stopping_threshold):
    # Initializing theta vector and learning rate
    num_features = X.shape[1] + 1
    theta = np.zeros(num_features)  # Initialize theta to zeros
    n = float(len(y))
    
    costs = []
    previous_cost = None

    # Add a column of ones to X for the intercept term
    X = np.column_stack((np.ones(X.shape[0]), X))

    # make y conform to the correct shape
    y = y.values.ravel()

    # Estimation of optimal parameters
    while True:
        # Making predictions
        y_predicted = np.dot(X, theta)
        
        # Calculating the current cost
        current_cost = mean_squared_error(y, y_predicted)
        
        # If the change in cost is less than or equal to stopping_threshold, stop gradient descent
        if previous_cost and abs(previous_cost - current_cost) <= stopping_threshold:
            break
        
        previous_cost = current_cost
        costs.append(current_cost)
        
        # Calculating the gradient
        gradient = -2/n * np.matmul(X.T, (y - y_predicted))
        
        # Updating theta
        theta -= learning_rate * gradient
     
    return theta, current_cost

# find new thetas using normal equations and calculate the new cost
def normal_equation(X, y, theta):
    # Add a column of ones to X for the intercept term
    X = np.column_stack((np.ones(X.shape[0]), X))

    # make y conform to the correct shape
    y = y.values.ravel()
    
    # find new theta using closed-form normal equations
    new_theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))

    # Making predictions
    y_predicted = np.dot(X, theta)
    y_new = np.dot(X, new_theta)
    
    # Calculating the current cost
    current_cost = mean_squared_error(y_predicted, y_new)

    return new_theta, current_cost

def normalize_features(X):
    # Subtract the mean value of each feature
    mean = np.mean(X, axis=0)
    normalized_X = X - mean

    # Scale the feature values by their respective standard deviations
    std_dev = np.std(X, axis=0)
    normalized_X /= std_dev

    return normalized_X

def main():
    # set up input file with columns
    clean_data_file()
    df = pd.read_csv('boston_cleaned.csv', sep='\s+')
    df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    x_nox_two = df[['DIS', 'RAD']]
    x_nox_all = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']]
    y_nox = df[['NOX']]
    x_medv_two = df[['AGE', 'TAX']]
    x_medv_all = df[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
    y_medv = df[['MEDV']]

    # 2a
    X_train, X_test, Y_train, Y_test = train_test_split(x_nox_two, y_nox, test_size=0.1, random_state=0)
    x_train_normalized = normalize_features(X_train)
    x_test_normalized = normalize_features(X_test)
    thetas_2a, squared_error_2a = training_gradient_descent(x_train_normalized, Y_train, 0.01, 1e-6)
    thetas_2a_normal, squared_error_2a_normal = normal_equation(x_test_normalized, Y_test, thetas_2a)

    # 2b
    X_train, X_test, Y_train, Y_test = train_test_split(x_nox_all, y_nox, test_size=0.1, random_state=0)
    x_train_normalized = normalize_features(X_train)
    x_test_normalized = normalize_features(X_test)
    thetas_2b,squared_error_2b = training_gradient_descent(x_train_normalized, Y_train, 0.01, 1e-6)

    # 2c
    X_train, X_test, Y_train, Y_test = train_test_split(x_medv_two, y_medv, test_size=0.1, random_state=0)
    x_train_normalized = normalize_features(X_train)
    x_test_normalized = normalize_features(X_test)
    thetas_2c, squared_error_2c = training_gradient_descent(x_train_normalized, Y_train, 0.01, 1e-6)
    thetas_2c_normal, squared_error_2c_normal = normal_equation(x_test_normalized, Y_test, thetas_2c)

    # 2d
    X_train, X_test, Y_train, Y_test = train_test_split(x_medv_all, y_medv, test_size=0.1, random_state=0)
    x_train_normalized = normalize_features(X_train)
    x_test_normalized = normalize_features(X_test)
    thetas_2d, squared_error_2d = training_gradient_descent(x_train_normalized, Y_train, 0.01, 1e-3)

    # print to output file
    print("PART I:", file=open('output.txt', 'a'))
    print("2A Thetas:\n", thetas_2a, "\nSquared error: ", squared_error_2a, file=open('output.txt', 'a'))
    print("\n2B Thetas:\n", thetas_2b, "\nSquared error: ", squared_error_2b, file=open('output.txt', 'a'))
    print("\n2C Thetas:\n", thetas_2c, "\nSquared error: ", squared_error_2c, file=open('output.txt', 'a'))
    print("\n2D Thetas:\n", thetas_2d, "\nSquared error: ", squared_error_2d, file=open('output.txt', 'a'))
    print("\n\nPART II: ", file=open('output.txt', 'a'))
    print("2A Thetas, using normal equation: \n", thetas_2a_normal, "\nSquared error: ", squared_error_2a_normal, file=open('output.txt', 'a'))
    print("\n2C Thetas, using normal equation: \n", thetas_2c_normal, "\nSquared error: ", squared_error_2c_normal, file=open('output.txt', 'a'))

    # remove cleaned-up input file
    if os.path.exists("boston_cleaned.csv"):
        os.remove("boston_cleaned.csv")

if __name__ == "__main__":
    main()
