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
import seaborn as sb
from scipy.ndimage import filters

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
        self.x_train_shrink = None                      # over and undersampled input
        self.y_train_shrink = None                      # over and undersampled target
        self.x_train_under =  None                      # undersampled input
        self.y_train_under =  None                      # undersampled target
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
        """ Process data for ease of use
        removes outliers, over and undersamples data, shuffles and standardizes data.
        """

#        #plotting outliers
#        ax = sb.boxplot(data=self.df_train, x='LABEL', y = 'FLUX.1')
#        ax.set(xlabel= "Class", ylabel = 'Flux for Feature[0]')
#        plt.title('First recorded flux for training stars')
#        print("plot saved!")
#        plt.savefig('../visuals/outliers.pdf')
#
#        #plotting example data
#        star_pos = self.x_train.iloc[35]
#        star_neg = self.x_train.iloc[50]
#        t = np.linspace(0,1920, len(star_pos))
#
#        fig, axs = plt.subplots(2, sharex= True)
#        axs[0].plot(t,star_pos)
#        axs[1].plot(t,star_neg)
#        plt.xlabel('Time[Hours]')
#        axs[0].set(ylabel = 'Flux', title = 'Exo-planet Star (#36)')
#        axs[1].set(ylabel = 'Flux', title =  'Non-exo-planet Star (#51)')
#        plt.savefig('../visuals/star_flux.pdf')
#        print("plot saved!")
#
        #removing outliers
        upper_outlier =  self.df_train[self.df_train['FLUX.1']>40000]
        self.df_train = self.df_train.drop((upper_outlier.index), axis=0)

#        lower_outlier =  self.df_train[self.df_train['FLUX.1']<-200000]
#        self.df_train = self.df_train.drop((lower_outlier.index), axis=0)

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


        # shrink dataset
        pos_index = self.y_train ==  1
        neg_index = self.y_train == 0
        pos_y = self.y_train[pos_index]
        pos_x = self.x_train[pos_index]

        neg_x = self.x_train[neg_index]
        neg_y = self.y_train[neg_index]

        index_choice = np.linspace(0,len(neg_y)-1, len(neg_y))
        random_index = (np.random.choice(index_choice, size= 400, replace = False)).astype(int)

        neg_x = neg_x.iloc[random_index,:]
        neg_y = neg_y[random_index]

        self.y_train_under =  np.concatenate([pos_y,neg_y])
        self.x_train_under =   np.concatenate([pos_x,neg_x])

        # Oversampling of undersampling
        sm = SMOTE(random_state=79)
        self.x_train_shrink, self.y_train_shrink = sm.fit_sample(self.x_train_under, self.y_train_under)


        # Shuffle training data
        idx = np.arange(len(self.y_train_over))
        np.random.shuffle(idx)
        self.y_train_over = self.y_train_over[idx]
        self.x_train_over = self.x_train_over.iloc[idx]

        idx = np.arange(len(self.y_train))
        np.random.shuffle(idx)

        self.y_train = self.y_train[idx]
        self.x_train = self.x_train.iloc[idx]

        idx = np.arange(len(self.y_train_shrink))
        np.random.shuffle(idx)

        self.y_train_shrink = self.y_train_shrink[idx]
        self.x_train_shrink = self.x_train_shrink[idx]


        # Standardize data
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)

        self.x_train_over = scaler.transform(self.x_train_over)

        self.x_train_shrink = scaler.transform(self.x_train_shrink)



        # Print analysis
        if self.print_results:


            labels = ['Train' , 'Test']
            pos_count = [count_train[1], count_test[1]]
            neg_count = [count_train[0], count_test[0]]

            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars


            fig, ax = plt.subplots(figsize = (5.5,3.5))
            bar1 = ax.bar(x - width/2, neg_count, width, label='No planet')
            bar2 = ax.bar(x + width/2, pos_count, width, label='Planet')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize = 14)

            for rect in bar1 + bar2:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

            plt.title('Data Distribution', fontsize = 16)
            plt.ylabel('Number of stars', fontsize = 14)
            plt.legend()
            plt.tight_layout()
            plt.savefig('../visuals/chart.pdf')
            print("plot saved!")

            print("\nData properties -------------")
            print("Train - outliers (rows, cols): ", self.df_train.shape)
            print("Test - outliers (rows, cols): ", self.df_test.shape)

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
