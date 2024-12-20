import numpy as np
import pandas as pd
import pickle

def compute_avg_f_score(output_hot, target):
    """
        Compute the Precision, Recall and F-Score for the predictions 'output_hot'.
    """    
     
    true_positives = np.sum((output_hot & target), axis = 1)
    false_positives = np.sum((output_hot & ~target), axis = 1)
    false_negatives = np.sum((~output_hot & target), axis = 1)
    
    denominator = true_positives + false_positives
    precision = np.where(denominator > 0,
                            true_positives / denominator,
                            np.zeros_like(true_positives)
                            )
    
    denominator = true_positives + false_negatives
    recall = np.where(denominator > 0,
                         true_positives / denominator,
                         np.zeros_like(true_positives)
                         )
    
    denominator = precision + recall
    f_score = np.where(denominator > 0,
                          2*(precision * recall) / denominator,
                          np.zeros_like(true_positives)
                          )
                          
    return np.mean(f_score), np.mean(precision), np.mean(recall)


def compute_avg_f_score_only(output_hot, target_hot):
    """
        Compute the Precision, Recall and F-Score for the predictions 'output_hot'.
    """
    output_hot = output_hot.values if isinstance(output_hot, pd.DataFrame) or isinstance(output_hot, pd.Series) else output_hot
    target_hot = target_hot.values if isinstance(target_hot, pd.DataFrame) or isinstance(target_hot, pd.Series) else target_hot
    
    f_score = 0
    for out, targ in zip(output_hot, target_hot):
        output_hot = np.asarray(out).astype(int)
        target = np.asarray(targ).astype(int)
        
        true_positives = np.sum((output_hot & target), axis = 1)
        false_positives = np.sum((output_hot & ~target), axis = 1)
        false_negatives = np.sum((~output_hot & target), axis = 1)
        
        denominator = true_positives + false_positives
        precision = np.where(denominator > 0,
                                true_positives / denominator,
                                np.zeros_like(true_positives)
                                )
        
        denominator = true_positives + false_negatives
        recall = np.where(denominator > 0,
                            true_positives / denominator,
                            np.zeros_like(true_positives)
                            )
        
        denominator = precision + recall
        f_score += np.where(denominator > 0,
                            2*(precision * recall) / denominator,
                            np.zeros_like(true_positives)
                            )
                          
    return f_score / len(output_hot)

def compute_avg_acuracy(y_hot, y_test):
    correct = np.sum(y_hot == y_test)
    return correct / y_test.size 
# Save dictionaries to a file
def save_params_to_file(file_name, **kwargs):
        with open(file_name, 'wb') as file:
            pickle.dump(kwargs, file)

def load_params_from_file(file_name):
    with open(file_name, 'rb') as file:
        loaded_params = pickle.load(file)
        params_trees = loaded_params['params_trees']
        params_forest = loaded_params['params_forest']
        params_knn = loaded_params['params_knn']
        params_rr = loaded_params['params_rr']
    return params_trees, params_forest, params_knn, params_rr

def load_features_from_file(file_name):
    with open(file_name, 'rb') as file:
        loaded_params = pickle.load(file)
        params_trees = loaded_params['features_trees']
        params_forest = loaded_params['features_forest']
        params_knn = loaded_params['features_knn']
        params_rr = loaded_params['features_rr']
    return params_trees, params_forest, params_knn, params_rr
