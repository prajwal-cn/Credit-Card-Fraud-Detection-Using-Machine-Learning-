import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load the credit card fraud detection dataset from a CSV file.
    """
    df = pd.read_csv(file_path)
    return df

def split_data(df, test_size=0.2):
    """
    Split the dataset into train and test sets.
    """
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

