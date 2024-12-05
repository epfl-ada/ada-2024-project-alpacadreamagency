from sklearn.linear_model import RidgeClassifier
import numpy as np
from ..utils.evaluation_utils import * 


def train_ridge_regression_model(X_train, y_train, X_test, y_test, classification_threshold, seed):

    model = RidgeClassifier(alpha=1.0, random_state=seed)
    model.fit(X_train, y_train)
    
    y_pred_logit = model.decision_function(X_test)
    exp_scores = np.exp(y_pred_logit - np.max(y_pred_logit, axis=1, keepdims=True))  # We do subtract max for numerical stability
    y_pred_logit = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    y_hot = (y_pred_logit > classification_threshold).astype(int)
    # y_pred = np.argmax(y_pred_logit, axis=1) 
    # y_hot = (y_pred == 1).astype(int).T
    
    f_score, precision, recall = compute_avg_f_score(y_hot, y_test)
    return model, y_hot, f_score, precision, recall