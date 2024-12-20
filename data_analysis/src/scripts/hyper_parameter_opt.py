# general imports
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# import for utility
from src.utils.data_utils import Settings as setting
from src.utils import general_utils as gt
from src.utils import evaluation_utils as eu
from src.utils.settings import Model_Settings, Settings

# import for scripts
from src.scripts.save_predictions import *

# import for models 
from src.plot_analysis.sentiment_analysis import *
from src.plot_analysis.theme_encoding import *
from src.models.crossvalidation import *
from src.models.neural_network import *

def eval_model(model, X_test, Y_test):
    y_pred_logit = model.predict_proba(X_test)
    # y_pred = (y_pred_logit > classification_threshold).astype(int)
    y_pred = np.argmax(y_pred_logit, axis=2) 
    y_hot = (y_pred == 1).astype(int).T
    
    f_score, precision, recall = eu.compute_avg_f_score(y_hot, Y_test)
    return f_score

def opt_model(model_funct, params, X_train, y_train, X_test, y_test):
    best_params = {}
    for parameter, values in params.items():
        print(f" · Optimizing with: {parameter}")
        best_fscore = -float("inf")
        best_value = None
        
        for value in values:
            model = model_funct(random_state=23, **{parameter: value})
            model.fit(X_train, y_train)
            fscore = eval_model(model, X_test, y_test)
            if fscore > best_fscore:
                best_fscore = fscore
                best_value = value
            print(f"   - Fscore = {fscore: .4f} Trained with: {parameter} = {value}.")
            
        best_params[parameter] = best_value
        print(f'Fscore: {best_fscore:.4f} with {best_params}')
    return best_params

def opt_trees(X_train, y_train, X_test, y_test):
    print("##########################")
    print("Training tress...")
    print("##########################")
    params = {
        "criterion": ["gini", "entropy", "log_loss"],  
        "splitter": ["best", "random"],               
        "max_depth": [None, 3, 5, 10, 20, 50],        
        "min_samples_split": [2, 5, 10, 20, 50],      
        "min_samples_leaf": [1, 2, 5, 10, 20],        
        "max_features": [None, "sqrt", "log2"],       
        "max_leaf_nodes": [None, 10, 20, 50, 100],   
        "ccp_alpha": [0.0, 0.01, 0.1, 1.0, 10.0],     
    }
    opt_model(DecisionTreeClassifier, params, X_train, y_train, X_test, y_test)
    

def opt_forest(X_train, y_train, X_test, y_test):
    print("##########################")
    print("Training forest...")
    print("##########################")
    params = {
        "n_estimators": [50, 100, 200, 500],           
        "criterion": ["gini", "entropy", "log_loss"],   
        "max_depth": [None, 3, 5, 10, 20, 50],          
        "min_samples_split": [2, 5, 10, 20],           
        "min_samples_leaf": [1, 2, 5, 10, 20],         
        "max_features": [None, "sqrt", "log2"],         
        "max_leaf_nodes": [None, 10, 20, 50, 100],     
        "bootstrap": [True, False],                   
        "class_weight": [None, "balanced", "balanced_subsample"], 
        "random_state": [42],                           
    }
    opt_model(RandomForestClassifier, params, X_train, y_train, X_test, y_test)
    
def opt_knn(X_train, y_train, X_test, y_test):
    print("##########################")
    print("Training KNN...")
    print("##########################")
    params = {
        "n_neighbors": [3, 5, 10, 20],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": [10, 20, 30, 50],
        "p": [1, 2],  # 1 for Manhattan distance, 2 for Euclidean
        "metric": ["minkowski", "euclidean", "manhattan", "chebyshev"], 
    }
    opt_model(KNeighborsClassifier, params, X_train, y_train, X_test, y_test)
    
def opt_ridge_reg(X_train, y_train, X_test, y_test):
    print("##########################")
    print("Training Ridge Regression...")
    print("##########################")
    params = {
        "penalty": ["l1", "l2", "elasticnet", None],
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [100, 200, 500, 1000],
        "fit_intercept": [True, False],
        "class_weight": [None, "balanced"], 
    }
    opt_model(LogisticRegression, params, X_train, y_train, X_test, y_test)
    
def opt_nn(X_train, y_train, X_test, y_test):
    # trained_model = start_train_model(training_columns, training_set, training_target_set, model_settings)
    # scores = test_model_get_score(trained_model, training_columns, testing_set, testing_target_set)
    ...
    
# Save dictionaries to a file
def save_params_to_file(file_name, **kwargs):
        with open(file_name, 'wb') as file:
            pickle.dump(kwargs, file)

def load_params_from_file():
    with open('optimized_params.pkl', 'rb') as file:
        loaded_params = pickle.load(file)
        params_trees = loaded_params['params_trees']
        params_forest = loaded_params['params_forest']
        params_knn = loaded_params['params_knn']
        params_rr = loaded_params['params_rr']
    return params_trees, params_forest, params_knn, params_rr

def main():
    training_columns = [
    'death', 'love', 'tragedy', 'violence', 'betrayal', 'friendship', 'happiness', 'fear', 'revenge', 'justice', 'hope', 'family', 
    'fate', 'greed', 'survival', 'transformation', 'Male actor count', 'Female actor count', 'N/A actor count', 'Character Count', 
    'Actors 0-20', 'Actors 20-30', 'Actors 30-40', 'Actors 40-60', 'Actors 60+', 'Character Count', 'revenue', 'runtime', 'release_year', 
    'vote_average', 'vote_count', 'adult', 'budget', 'popularity', 'sentiment'
    ]
    target_column = "genre_hot"
    MOVIES = pd.read_csv("cleaned_data.csv")

    model_settings = Model_Settings()

    X_train, y_train, X_test, y_test, train_full, test_full = train_and_test_split(
        MOVIES, 
        training_columns, target_column, 
        train_proportion =  model_settings.TEST_PROPORTION,
        seed = model_settings.SEED
    )


    params_trees = opt_trees(X_train, y_train, X_test, y_test)
    params_forest = opt_forest(X_train, y_train, X_test, y_test)
    params_knn = opt_knn(X_train, y_train, X_test, y_test)
    params_rr = opt_ridge_reg(X_train, y_train, X_test, y_test)
    # opt_nn()    

    # Save all parameter dictionaries
    save_params_to_file(
        'optimized_params.pkl', 
        params_trees=params_trees, 
        params_forest=params_forest, 
        params_knn=params_knn, 
        params_rr=params_rr
    )
    
    print(*load_params_from_file())

if __name__ == "__main__":
    main()