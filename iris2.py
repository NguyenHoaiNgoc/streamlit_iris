from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train, y_train)
    
    return rf_clf

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    X, y = load_data()
    model = train_model(X, y)
    save_model(model, "iris_model.pkl")