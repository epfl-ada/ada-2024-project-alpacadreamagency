# general imports
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from src.models import neural_network as nn
# import for utility
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

def eval_model(model, X_test, Y_test, use_proba, class_threshold = 0.03):
    if use_proba: 
        y_pred_logit = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_logit, axis=2) 
        y_hot = (y_pred == 1).astype(int).T
    else:
        y_pred_logit = model.decision_function(X_test)
        exp_scores = np.exp(y_pred_logit - np.max(y_pred_logit, axis=1, keepdims=True))  # We do subtract max for numerical stability
        y_pred_logit = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        y_hot = (y_pred_logit > class_threshold).astype(int)
        
    
    f_score, precision, recall = eu.compute_avg_f_score(y_hot, Y_test)
    return f_score

def opt_model_params(
    model_funct, params, X_train, y_train, X_test, y_test, 
    random_state = None, use_proba = True, class_threshold = 0.03
):
    best_params = {}
    for parameter, values in params.items():
        print(f" · Optimizing with: {parameter}")
        best_fscore = -float("inf")
        best_value = None
        
        for value in values:
            if random_state is not None:
                model = model_funct(random_state=random_state, **{parameter: value})
            else:
                model = model_funct(**{parameter: value})
            model.fit(X_train, y_train)
            fscore = eval_model(model, X_test, y_test, use_proba, class_threshold)
            if fscore > best_fscore:
                best_fscore = fscore
                best_value = value
            print(f"   - Fscore = {fscore: .4f} Trained with: {parameter} = {value}.")
            
        best_params[parameter] = best_value
        print(f'Fscore: {best_fscore:.4f} with {best_params}')
    return best_params

def opt_model_features(model_funct, params, X_train, y_train, X_test, y_test, all_features, random_state = None, use_proba = True):
    exclude_features = []
    
    if random_state is not None:
        model = model_funct(random_state=random_state, **params)
    else:
        model = model_funct(**params)
        
    model.fit(X_train, y_train)
    ref_fscore = eval_model(model, X_test, y_test, use_proba)
    print(f"Reference f_score = {ref_fscore:.4f} with {model_funct}")
    
    for feature in all_features:
        X_train_drop = X_train.drop(columns = [feature])
        X_test_drop  = X_test.drop(columns = [feature])
        
        print(f" · Optimizing with: {feature}")
        if random_state is not None:
            model = model_funct(random_state=random_state, **params)
        else:
            model = model_funct(**params)
        model.fit(X_train_drop, y_train)
        fscore = eval_model(model, X_test_drop, y_test, use_proba)
        if fscore >= ref_fscore:
            exclude_features += [feature]
        print(f"   - Fscore = {fscore: .4f} Excluding: {feature}.")
            
    return exclude_features

def opt_param_trees(X_train, y_train, X_test, y_test):
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
    return opt_model_params(DecisionTreeClassifier, params, X_train, y_train, X_test, y_test, random_state=42)    

def opt_param_forest(X_train, y_train, X_test, y_test):
    print("##########################")
    print("Training forest...")
    print("##########################")
    params = {
        "n_estimators": [10, 50, 100, 200],           
        "criterion": ["gini", "entropy", "log_loss"],   
        "max_depth": [None, 1, 5, 20, 50],          
        "max_features": [None, "sqrt", "log2"],         
        "max_leaf_nodes": [None, 50, 100, 200],     
        "bootstrap": [True, False],                   
    }
    return opt_model_params(RandomForestClassifier, params, X_train, y_train, X_test, y_test, random_state=42)
    
def opt_param_knn(X_train, y_train, X_test, y_test):
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
    return opt_model_params(KNeighborsClassifier, params, X_train, y_train, X_test, y_test)
    
def opt_param_ridge_reg(X_train, y_train, X_test, y_test):
    print("##########################")
    print("Training Ridge Regression...")
    print("##########################")
    params = {
        "alpha": [0.1, 1.0, 10.0, 100.0], 
        "fit_intercept": [True, False],  
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"], 
        "positive": [True, False],  
    }
    return opt_model_params(RidgeClassifier, params, X_train, y_train, X_test, y_test, random_state=42, use_proba=False)
    
def opt_nn(X_train, y_train, X_test, y_test, training_columns, model_settings):
    for shape in [True, False]:
        model_settings = Model_Settings(DENSE_SHAPE=shape)
        
        trained_model, histories, model_name = nn.start_train_model(training_columns, X_train, y_train, X_test, y_test, model_settings)
        
        scores = nn.test_model_get_score(trained_model, training_columns, X_test, y_test)
        ref_fscore = scores[1][0]
        
        exclude_features = []
        for feature in training_columns:
            train_drop_columns = list(training_columns)
            train_drop_columns = train_drop_columns.remove(feature)
            
            trained_model, histories, model_name = nn.start_train_model(train_drop_columns, X_train, y_train, X_test, y_test, model_settings)
            
            scores = nn.test_model_get_score(trained_model, train_drop_columns, X_test, y_test)
            fscore = scores[1][0]
            if fscore >= ref_fscore:
                exclude_features += [feature]
            print(f"   - Fscore = {fscore: .4f} Excluding: {feature}.")
            
        print(f"For {shape = }, exclude: {exclude_features}")
    

def parameter_opt():
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
    params_trees = opt_param_trees(X_train, y_train, X_test, y_test)
    # params_trees =  {'criterion': 'gini', 'splitter': 'best', 'max_depth': 10, 'min_samples_split': 50, 'min_samples_leaf': 10, 'max_features': None, 'max_leaf_nodes': None, 'ccp_alpha': 0.0}
    params_forest = opt_param_forest(X_train, y_train, X_test, y_test)
    # params_forest = {'n_estimators': 200, 'criterion': 'gini', 'max_depth': 20, 'max_features': None, 'max_leaf_nodes': None, 'bootstrap': False}
    params_knn = opt_param_knn(X_train, y_train, X_test, y_test)
    # params_knn = {'n_neighbors': 3, 'weights': 'distance', 'algorithm': 'auto', 'leaf_size': 10, 'p': 1, 'metric': 'manhattan'}
    params_rr = opt_param_ridge_reg(X_train, y_train, X_test, y_test)
    # params_rr = {'alpha': 10.0, 'fit_intercept': True, 'solver': 'auto', 'positive': False}
    # opt_nn(X_train, y_train, X_test, y_test, training_columns, model_settings)    

    file_name = 'optimized_params.pkl'
    # Save all parameter dictionaries
    eu.save_params_to_file(
        file_name, 
        params_trees=params_trees, 
        params_forest=params_forest, 
        params_knn=params_knn, 
        params_rr=params_rr
    )
    
    print(*eu.load_params_from_file(file_name))

def feature_opt():
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
    
    file_params_name = 'optimized_params.pkl'
    params_trees, params_forest, params_knn, params_rr = eu.load_params_from_file(file_params_name)
    features_for_trees =  opt_model_features(DecisionTreeClassifier, params_trees, X_train, y_train, X_test, y_test, training_columns, random_state=42)
    # features_for_forest = opt_model_features(RandomForestClassifier, params_forest, X_train, y_train, X_test, y_test, training_columns, random_state=42)
    features_for_forest = [
        'death','love','tragedy','betrayal','fear','justice','fate','survival', 'Male actor count', 'Character Count','Actors 0-20', 'Actors 60+', 'Character Count', 'adult', 'budget', 'popularity'
    ]
    features_for_knn =    opt_model_features(KNeighborsClassifier, params_knn, X_train, y_train, X_test, y_test, training_columns)
    features_for_rr =     opt_model_features(RidgeClassifier, params_rr, X_train, y_train, X_test, y_test, training_columns, random_state=42, use_proba=False)


    file_feature_name = 'optimized_features.pkl'
    eu.save_params_to_file(
        file_feature_name, 
        features_trees=features_for_trees, 
        features_forest=features_for_forest, 
        features_knn=features_for_knn, 
        features_rr=features_for_rr
    )
    
    print(*eu.load_features_from_file(file_feature_name))

if __name__ == "__main__":


    parameter_opt()
    
    feature_opt()