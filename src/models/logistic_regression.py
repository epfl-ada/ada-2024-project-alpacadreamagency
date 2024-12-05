from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification


def train_lr_model(data_train, data_test, classification_threshold):
    X_train, y_train = data_train
    X_test, y_test = data_test
    model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
