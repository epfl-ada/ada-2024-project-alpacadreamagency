from sklearn.linear_model import LogisticRegression

def train_lr_model(X_train, y_train, X_test, y_test, classification_threshold):

    model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred_logit = model.predict_proba(X_test)
    y_pred = (y_pred_logit > classification_threshold).astype(int)
    
    return model, y_pred
