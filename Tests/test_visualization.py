import pandas as pd
import unittest
import seaborn as sns
import matplotlib.pyplot as plt
from src.data import load_data
from src.visualization import plot_histograms, plot_correlation_matrix

class TestVisualization(unittest.TestCase):

    def setUp(self):
        # Load the data
        data_file = 'data/creditcard.csv'
        self.df = load_data(data_file)

    def test_plot_histograms(self):
        # Create a sample histogram plot
        fig, axs = plt.subplots(2, 2)
        plot_histograms(self.df, axs)
        plt.close()

        # Check the number of subplots
        self.assertEqual(len(axs), 2)
        self.assertEqual(len(axs[0]), 2)
        self.assertEqual(len(axs[1]), 2)

    def test_plot_correlation_matrix(self):
        # Create a sample correlation matrix plot
        fig, ax = plt.subplots()
        plot_correlation_matrix(self.df, ax)
        plt.close()

        # Check the plot title
        self.assertEqual(ax.get_title(), 'Credit Card Transactions: Correlation Matrix')

