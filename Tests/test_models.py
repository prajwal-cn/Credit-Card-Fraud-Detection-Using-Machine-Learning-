import pandas as pd
import unittest
import os
import pickle
from src.data import load_data, split_data
from src.models import train_model, save_model

class TestModels(unittest.TestCase):

    def setUp(self):
        # Load the data
        data_file = 'data/creditcard.csv'
        self.df = load_data(data_file)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.df)

        # Train the model
        self.model = train_model(self.X_train, self.y_train)

        # Save the model
        self.model_file = 'models/test_model.pkl'
        save_model(self.model, self.model_file)

    def tearDown(self):
        # Remove the test model file
        os.remove(self.model_file)

    def test_save_model(self):
        # Load the saved model
        with open(self.model_file, 'rb') as f:
            loaded_model = pickle.load(f)

        # Check the model's predictions on the test set
        y_pred = loaded_model.predict(self.X_test)
        self.assertEqual(y_pred.shape, self.y_test.shape)

