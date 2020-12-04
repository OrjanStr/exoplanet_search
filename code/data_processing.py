# Import essensial packages
import pandas as pd
import numpy as np
import os
from IPython.display import display
import matplotlib.pyplot as plt

class process_data:
    def __init__(self):
        self.df     = None                  # Entire dataset with labels
        self.target = None                  # Target labels
        self.input  = None                  # Dataset
        self.path   = "../data/heart.csv"   # Path of file containing data

    def read_data(self):
        self.df     = pd.read_csv(self.path)
        self.input  = self.df.copy()
        self.target = self.input.pop('target')
        return 'sha'

df = process_data()
df.read_data()
