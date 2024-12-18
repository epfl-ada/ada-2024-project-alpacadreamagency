import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import Counter
import random
import torch
from sklearn.model_selection import train_test_split


def cross_validation_split(data, training_columns, testing_column, set_to_test, train_proportion = 0.2, seed = 42):
    """
        Split the data into training and testing. Return the 'set_to_test' set out of all the possible splits
        'set_to_test' goes from 0 to 4 -> 5 possible Splits. 
    """    
    
    divisions = int(1 / train_proportion)
    if set_to_test is None or set_to_test > divisions or set_to_test < 0:
        print("BE CAREFUL, ONLY VALUES FORM 0 to 1. Setting testing set to 0")
        set_to_test = 0
    # GET THE MOST FROM THE UNBALANCED CLASS
    train_proportion = train_proportion # 1/train_proportion divisioins, as default 2
    n = np.round(train_proportion, 0).astype(int)
        
    data_suffled = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    training_set = data_suffled[:n*set_to_test] + data_suffled[n*(set_to_test+1):] 
    testing_set = data_suffled[n*set_to_test:n*(set_to_test+1)]

    return (
        training_set[training_columns], 
        training_set[testing_column], 
        testing_set[training_columns], 
        testing_set[testing_column], 
    )

def train_and_test_split(data, training_columns, testing_column, train_proportion = 0.2, seed = 42):
    """
        Bultin random data split into train and test.
        (Used for this milestone)
    """
    
    training_set, testing_set = train_test_split(data, test_size=train_proportion, random_state=seed)

    return (
        training_set[training_columns], 
        training_set[testing_column], 
        testing_set, 
        testing_set[testing_column], 
    )