import pandas as pd
import unittest
from src.data import load_data, split_data

class TestData(unittest.TestCase):

    def test_load_data(self):
        # Load the data
        data_file = 'data/creditcard.csv'
        df = load_data(data_file)

        # Check the data shape
        expected_shape = (284807, 31)
        self.assertEqual(df.shape, expected_shape)

    def test_split_data(self):
        # Create a sample dataframe
        df = pd.DataFrame({'X': [1, 2, 3, 4], 'y': [0, 0, 1, 1]})

        # Split the data
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.5)

        # Check the split shapes
        expected_train_shape = (2, 1)
        expected_test_shape = (2, 1)
        self.assertEqual(X_train.shape, expected_train_shape)
        self.assertEqual(X_test.shape, expected_test_shape)
        self.assertEqual(y_train.shape, (2,))
        self.assertEqual(y_test.shape, (2,))

