from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from ..utils.evaluation_utils import * 


def train_knn_model(X_train, y_train, X_test, y_test, classification_threshold, n_neighbors):

    model = KNeighborsClassifier(n_neighbors=n_neighbors,  n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_logit = model.predict_proba(X_test)
    # y_pred = (y_pred_logit > classification_threshold).astype(int)
    y_pred = np.argmax(y_pred_logit, axis=2) 
    y_hot = (y_pred == 1).astype(int).T
    
    f_score, precision, recall = compute_avg_f_score(y_hot, y_test)
    return model, y_hot, f_score, precision, recall