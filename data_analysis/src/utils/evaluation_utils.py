import numpy as np
import pandas as pd
import ast

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

