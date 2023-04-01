from sklearn.ensemble import RandomForestClassifier
import pickle

def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier on the training set.
    """
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    return rf

def save_model(model, file_path):
    """
    Save the trained model as a pickle file.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

