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

def preprocess():
    diamonds_df = pd.read_csv("diamonds.csv")
    
    print("======================Task1======================")
    print("Columns: ", diamonds_df.columns)
    print(diamonds_df.head())
    
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
    
    # This goes through each cut, color and clarity combinations
    for cut in diamonds_df['cut'].unique():
        for color in diamonds_df['color'].unique()   :
            for clarity in diamonds_df['clarity'].unique():
                # Example of how to do multiple columns equaling something
                # dataframe[ ( dataframe['column'] == value ) & ( dataframe['column'] == value ) ]
                
                 # This gets the number of datapoints for this combination
                no_dp = len(diamonds_df[(diamonds_df['cut'] == cut) & 
                                        (diamonds_df['color'] == color) &
                                        (diamonds_df['clarity'] == clarity)])
                
                # Only print the datapoints with 801+ values (more than 800)
                if (800 < no_dp):
                    print(cut, " : ", color, " : ", clarity)
                    
     
    # From now on all processing will be on these subsets 
    # corresponding to the various different grades 
    # (e.g. Machine Learning Assignment 3: Regression & 
    # optimisation ('Ideal', 'E', 'VS2')). 
    
    # Going grade-by-grade split the data-points into 
    # features [1 point] and targets [1 point]. 
    
    # Use the carat, depth and table value as 
    # features and the selling price as target.
    
    # Create a loop going over all combinations of cut, colour,
    # and clarity [1 point] and count the number of 
    # data-points within each subset [1 point]. 
    
    # Select only the datasets containing more than 800 
    # data-points for further processing [1 point].


    

def main():
    preprocess()
    

main()


