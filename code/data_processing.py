# Import essensial packages
import pandas as pd
import numpy as np
import os
from IPython.display import display
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

class process_data:
    def __init__(self, print_results=True):
        """
        For reading and processing exoplanet data.
        Arguments:
            print_results (Boolean, Default = True): Whether or not to print analysis results
         """
        self.df_train       = None                      # Entire training dataset with labels
        self.df_test        = None                      # Entire testing dataset with labels
        self.x_train        = None                      # Training input
        self.x_test         = None                      # Test input
        self.y_train        = None                      # Training target
        self.y_test         = None                      # Test target
        self.x_train_over   = None                      # Oversampled input
        self.y_train_over   = None                      # Oversampled target
        self.input          = None                      # Dataset
        self.path_train     = "../data/exoTrain.csv"    # Path of file containing training data
        self.path_test      = "../data/exoTest.csv"     # Path of file containing test data
        self.print_results  = print_results             # Print analysis

        # Read and process data
        self.read_data()
        self.process()

    def read_data(self):
        """ Read the two csv files and store into train and test """
        self.df_train   = pd.read_csv(self.path_train)
        self.x_train    = self.df_train.copy()
        self.y_train    = self.x_train.pop('LABEL')

        self.df_test    = pd.read_csv(self.path_test)
        self.x_test     = self.df_test.copy()
        self.y_test     = self.x_test.pop('LABEL')


    def process(self):
        """ Process data for ease of use """
        # How many positive/negative labels?
        count_train = self.y_train.value_counts().values
        count_test  = self.y_test.value_counts().values

        # Oversampling
        sm = SMOTE(random_state=42)
        self.x_train_over, self.y_train_over = sm.fit_sample(self.x_train, self.y_train)
        over_count = self.y_train_over.value_counts().values

        # Convert labels to one hot
        Encoder = LabelEncoder()
        self.y_train        = Encoder.fit_transform(self.y_train)
        self.y_test         = Encoder.fit_transform(self.y_test)
        self.y_train_over   = Encoder.fit_transform(self.y_train_over)

        # Shuffle training data
        idx = np.arange(len(self.y_train_over))
        np.random.shuffle(idx)
        self.y_train_over = self.y_train_over[idx]
        self.x_train_over = self.x_train_over.iloc[idx]

        idx = np.arange(len(self.y_train))
        np.random.shuffle(idx)
        self.y_train = self.y_train[idx]
        self.x_train = self.x_train.iloc[idx]

        # Standardize data
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

        # Print analysis
        if self.print_results:
            print("\nData properties -------------")
            print("Train (rows, cols): ", self.df_train.shape)
            print("Test (rows, cols): ", self.df_test.shape)

            print("\nMissing Data:")
            print("Tot missing data in train: ", self.df_train.isnull().sum().sum())
            print("Tot missing data in test: ", self.df_test.isnull().sum().sum())

            print("\nLabel distribution before oversampling:")
            print("Exoplanets in train: %i/%i, %.3f%%" %(count_train[1], count_train.sum(), 100*count_train[1]/count_train.sum()))
            print("Exoplanets in test: %i/%i, %.3f%%" %(count_test[1], count_test.sum(), 100*count_test[1]/count_test.sum()))

            print("\nLabel distribution after oversampling:")
            print("Exoplanets in train: %i/%i, %.3f%%" %(over_count[1], over_count.sum(), 100*over_count[1]/over_count.sum()))


if __name__ == "__main__":
    df = process_data()
