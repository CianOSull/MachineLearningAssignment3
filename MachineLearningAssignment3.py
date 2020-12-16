# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:54:03 2020

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

def num_coefficients_3(d):
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        t = t+1
    return t

# def calculate_model_function(deg,data, p):
#     result = np.zeros(data.shape[0])    
#     k=0
#     for n in range(deg+1):
#         for i in range(n+1):
#             result += p[k]*(data[:,0]**i)*(data[:,1]**(n-i))
#             k+=1
#     return result

def eval_poly_3(d,a,x,y,z):
    r = 0
    t = 0
    for n in range(d+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        r += a[t]*(x**i)*(y**j)*(z**k)
                        t = t+1
    return r

# P is parameter vector
def calculate_poly_function(deg, data, p):
    r = 0
    t = 0
    for n in range(deg+1):
        for i in range(n+1):
            for j in range(n+1):
                for k in range(n+1):
                    if i+j+k==n:
                        r += p[t]*(data[:,0]**i)*(data[:,1]**j)*(data[:,2]**k)
                        t = t+1
    return r
    

def linearize(deg,data, p0):
    f0 = calculate_poly_function(deg,data,p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_poly_function(deg,data,p0)
        p0[i] -= epsilon
        di = (fi - f0)/epsilon
        J[:,i] = di
    return f0,J
    
def calculate_update(y,f0,J):
    l=1e-2
    N = np.matmul(J.T,J) + l*np.eye(J.shape[1])
    r = y-f0
    n = np.matmul(J.T,r)    
    dp = np.linalg.solve(N,n)       
    return dp

def regression(deg, data, target):
    max_iter = 10
    p0 = np.zeros(num_coefficients_3(deg))
    for i in range(max_iter):
        f0,J = linearize(deg,data, p0)
        dp = calculate_update(target,f0,J)
        p0 += dp
    
    return p0


def main():
    feature_df_list, target_df_list = Preprocess()
    
    # Task 2 on Wards
    print("======================Task2-5======================")
    
    # Task 6
    kf = model_selection.KFold(n_splits=2, shuffle=True)
    
    best_results = []
    best_degrees = []
    best_p0 = []
    
    # For each dataset
    for index in range(len(feature_df_list)):
        best_deg_results = []
        best_p0_deg_results = []
        best_deg_difference = -1e-1000
        
        # for degress
        for deg in range(4):
            # K fold
            difference_list = []
            p0_list = []
            best_difference = -1e-1000
            
            for train_index, test_index in kf.split(feature_df_list[index]):
    
                p0 = regression(deg, feature_df_list[index][train_index], target_df_list[index][train_index])
                
                prediction = calculate_poly_function(deg, feature_df_list[index][test_index], p0)
                
                # Find the mean difference between the price estimates and 
                # difference = np.mean(prediction) - np.mean(p0)
                difference = np.mean(target_df_list[index] - np.mean(prediction))
                difference_list.append(difference)
                p0_list.append(p0)

                
            for i in range(len(difference_list)):
                if best_difference > difference_list[i]:
                    best_difference = difference_list[i]
                    
            best_deg_results.append(best_difference)
            best_p0_deg_results.append(p0_list[difference_list.index(best_difference)])
            
        for i in range(len(best_deg_results)):
            if best_deg_difference > best_deg_results[i]:
                best_deg_difference = best_deg_results[i]
                
        best_results.append(best_deg_difference)
        best_degrees.append(best_deg_results.index(best_deg_difference))
        best_p0.append(best_p0_deg_results[best_deg_results.index(best_deg_difference)])
    
    # print(best_results)
    print("The best degrees for each dataset are: ", best_degrees)  
    # print("The best p0 for each dataset are: ", best_p0)  
    
    # Task 7
    # true prices and price estimates
    # true prices on x axis
    # price estimates on y axis
    # If you plot this, they should be going along the diagonal
    # rather than all of the place
    plt.close("all")
    
    for index in range(len(best_p0)):
        
        prediction = calculate_poly_function(best_degrees[index], feature_df_list[index], best_p0[index])

        plt.figure()
        plt.scatter(prediction, target_df_list[index], color ='g')
        plt.title("True Prices and Estimated Prices for Dataset " + str(index+1))
        plt.xlabel("True Prices")
        plt.ylabel("Price Estimates")
        labels = ['Data Points']
        plt.legend(labels, loc="upper left", title="Data Points")
    
main()
