import numpy as np

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

