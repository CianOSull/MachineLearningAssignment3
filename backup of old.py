# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 16:13:06 2020

@author: Cian
"""




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neural_network
from sklearn import model_selection
from sklearn import metrics

# carat
# depth
# table
# selling price
# assignment data points and features maybe?
# Somehting about groups and then work on groups
# Combinations is groups

# two components are used in lab so it is binominal i think
# degree in a univariete polynominal degree is teh highest d until you sum
# basically the amount of numbers you sum think?
# No using functions for polnominal functions in lab because 
# librareis might need other stuff so we make our own
# Look up coefficients from lectures because I have no idea why it exists

# Task 1
def Preprocess():
    diamonds_df = pd.read_csv("diamonds.csv")
    
    print("======================Task1======================")
    # print("Columns: ", diamonds_df.columns)
    # print(diamonds_df.head())
    # This is just a demonstration of how values can be used to make numpy arrays
    # print("Values: ", diamonds_df[['carat', 'depth', 'table']].values, type(diamonds_df[['carat', 'depth', 'table']].values))
    # print("Values: ", diamonds_df['price'].values, type(diamonds_df['price'].values))
    
    # Create a function that loads the file and extracts what 
    # types of cut qualities [1 point], colour grades [1 point], 
    # and clarity grades [1 point] are represented. 
    
    # List all the unique values of the cut column
    print("Cut: ", diamonds_df['cut'].unique())
    
    # List all the unique values of the color column
    print("Color: ", diamonds_df['color'].unique())
    
    # List all the unique values of the clarity column
    print("Clarity: ", diamonds_df['clarity'].unique())
    
    # For each combination of these cut, colour and clarity 
    # grades extract the corresponding data-points [1 point].
    # From now on all processing will be on these subsets 
    # corresponding to the various different grades 
    # (e.g. Machine Learning Assignment 3: Regression & 
    # optimisation ('Ideal', 'E', 'VS2')). 
    
    # Create lists taht will contain the optimal cobminations
    cut_list = []
    color_list = []
    clarity_list = []
    
    feature_df_list = []
    target_df_list = []
    
    # Create a loop going over all combinations of cut, colour,
    # and clarity [1 point] and count the number of 
    # data-points within each subset [1 point]. 
    # This goes through each cut, color and clarity combinations
    for cut in diamonds_df['cut'].unique():
        for color in diamonds_df['color'].unique()   :
            for clarity in diamonds_df['clarity'].unique():
                # Example of how to do multiple columns equaling something
                # dataframe[ ( dataframe['column'] == value ) & ( dataframe['column'] == value ) ]
                
                # Create new dataframe using the combinations
                df = diamonds_df[(diamonds_df['cut'] == cut) & 
                                 (diamonds_df['color'] == color) &
                                 (diamonds_df['clarity'] == clarity)]
                
                # Select only the datasets containing more than 800 
                # data-points for further processing [1 point].
                # Only print the datapoints with 801+ values (more than 800)
                # Also add the combinations to the list
                if (800 < len(df)):
                    print(cut, " : ", color, " : ", clarity)
                                        
                    # Going grade-by-grade split the data-points into 
                    # features [1 point] and targets [1 point]. 
                    # Use the carat, depth and table value as 
                    # features and the selling price as target.
                    
                    # Extract columns
                    # df = df[['column', 'column']]
                    # Add the combinations to a list of all the dfs
                    feature_df_list.append(df[['carat', 'depth', 'table']].values)
                    target_df_list.append(df['price'].values)
                    
    # print(feature_df_list[0])
        
    return feature_df_list, target_df_list

# Task 2
# Example coefficants of 3 from lab 9
def num_coefficients_3(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t = t+1
    return t

# polynomial example of 3 from lab 9
# This does the prediction
# P is parameter vector
def calculate_model_function(d, a, data):    
    print(data)
    r = 0
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        # r += a[t]*(x**i)*(y**j)*(z**k)
                        # r += parameter_vector[t] * (feature_points[:, 0]  i) * (feature_points[:, 1]  j) * (feature_points[:, 2] ** k)
                       
                        r += a[t]*(data[:,0]**i)*(data[:,1]**j)*(data[:,2]**k)
                        # p[k]*(data[:,0]**i)*(data[:,1]**(n-i))
                        t = t+1
   
    return r

def linearize(deg,data, p0):
    f0 = calculate_model_function(deg, p0, data)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, p0, data)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
    return f0,J

# Task 4
# J is the jackovian matrix
def calculate_update(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)       
    return dp


# Task 5 
def regression(degrees, data, target):
    max_iter = 1
    for deg in range(5):
        p0 = np.zeros(num_coefficients_3(deg))
        for i in range(max_iter):
            f0,J = linearize(deg, p0, data)
            dp = calculate_update(f0, target, J)
            # print("regression: p0\n", p0, "\n")
            # print("regression: dp\n", dp, "\n")
            # p0 += dp
    
    return p0
   
# Task 7
# true prices and price estimates
# true prices on x axis
# price estimates on y axis
# If you plot this, they should be going along the diagonal
# rather than all of the place    

def main():
    feature_df_list, target_df_list = Preprocess()
    
    # print(feature_df_list[0])
    # print("Column1: ", feature_df_list[0][0])
    # print("Value1: ", feature_df_list[0][0][0])
    # Seperate them, each in lsit is a unique dataset
    # data = feature_df_list[0]

    # Task 6
    kf = model_selection.KFold(n_splits=4, shuffle=True)
    
    for index in range(len(feature_df_list)):
        for train_index, test_index in kf.split(feature_df_list[index]):
            # print(feature_df_list[index][train_index])
            # print(feature_df_list[index][train_index])
            
            p0 = regression(4, feature_df_list[0][train_index], target_df_list[index][train_index])
            
            
            # print(p0)
            # Only do it for one of the datasets for testing
            break
        break
    
    # def calculate_model_function(deg, paramaterVector, column1,column2,column3):
    
    # calculate_model_function(3, , feature_df_list[0][0][0], feature_df_list[0][0][1], feature_df_list[0][0][2])
    
main()


