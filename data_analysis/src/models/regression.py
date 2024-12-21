from sklearn.linear_model import RidgeClassifier
import numpy as np
from ..utils.evaluation_utils import * 
from ..utils.settings import Model_Settings

def train_ridge_regression_model(X_train, y_train, X_test, y_test, model_settings: Model_Settings, params, exclude_fet):
    X_train_drop = X_train.drop(columns = exclude_fet)
    X_test_drop = X_test.drop(columns = exclude_fet)
    
    
    model = RidgeClassifier(random_state=model_settings.SEED, **params)
    model.fit(X_train_drop, y_train)
    
    y_pred_logit = model.decision_function(X_test_drop)
    exp_scores = np.exp(y_pred_logit - np.max(y_pred_logit, axis=1, keepdims=True))  # We do subtract max for numerical stability
    y_pred_logit = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    y_hot = (y_pred_logit > model_settings.THRESHOLD_RIDGE).astype(int)
    # y_pred = np.argmax(y_pred_logit, axis=1) 
    # y_hot = (y_pred == 1).astype(int).T
    
    f_score, precision, recall = compute_avg_f_score(y_hot, y_test)
    acuracy = compute_avg_acuracy(y_hot, y_test)
    return model, y_hot, acuracy, f_score, precision, recall