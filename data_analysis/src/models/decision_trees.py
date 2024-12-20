from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ..utils.evaluation_utils import * 
from ..utils.settings import Model_Settings


def train_tree_model(X_train, y_train, X_test, y_test, model_settings: Model_Settings, params, exclude_fet):
    X_train_drop = X_train.drop(columns = exclude_fet)
    X_test_drop = X_test.drop(columns = exclude_fet)
    
    model = DecisionTreeClassifier(random_state=model_settings.SEED, **params)
    model.fit(X_train_drop, y_train)
    
    y_pred_logit = model.predict_proba(X_test_drop)
    # y_pred = (y_pred_logit > classification_threshold).astype(int)
    y_pred = np.argmax(y_pred_logit, axis=2) 
    y_hot = (y_pred == 1).astype(int).T
    
    f_score, precision, recall = compute_avg_f_score(y_hot, y_test)
    acuracy = compute_avg_acuracy(y_hot, y_test)
    return model, y_hot, acuracy, f_score, precision, recall

def train_random_forest_model(X_train, y_train, X_test, y_test, model_settings: Model_Settings, params, exclude_fet):
    X_train_drop = X_train.drop(columns = exclude_fet)
    X_test_drop = X_test.drop(columns = exclude_fet)
    
    model = RandomForestClassifier(random_state=model_settings.SEED, n_jobs=-1, **params)
    model.fit(X_train_drop, y_train)
    
    y_pred_logit = model.predict_proba(X_test_drop)
    # y_pred = (y_pred_logit > classification_threshold).astype(int)
    y_pred = np.argmax(y_pred_logit, axis=2) 
    y_hot = (y_pred == 1).astype(int).T
    
    acuracy = compute_avg_acuracy(y_hot, y_test)
    f_score, precision, recall = compute_avg_f_score(y_hot, y_test)
    return model, y_hot, acuracy, f_score, precision, recall

