# Essensial libraries
import numpy as np
import matplotlib.pyplot as plt

# Processing and metrics
from sklearn                  import metrics

# Models
from sklearn.svm           import SVC, LinearSVC

# Self made classes
import sys
sys.path.insert(0,"..")
from data_processing import process_data
from model_setup import evaluate_model

def tune_SVM():
    """
    Finds best parameter values for SVM using grid search and evaluate model
    """
    model1 = LinearSVC()
#C=  0.01 gamma=  0.00020691380811147902
#C=  0.01 gamma=  0.00069519279617756050.008858667904100823
#C=  0.008858667904100823 gamma=  0.00029763514416313193
    c_lst = np.logspace(-3,2,10)
    kernel_lst = ['linear','poly', 'sigmoid','rbf']
    gamma_lst = np.logspace(-1,0,10)

    for i in range(10):

            print (c_lst[i])
            model = SVC(kernel ='sigmoid', gamma = 0.36 ,C = c_lst[i], probability = True )
            evaluate_model(model, x_train_down, x_test, y_train_down, y_test, 'baseline_CM_SVM', 'Baseline confusion matrix: SVM  ', 'svm_rp_name', 'svm_cm_name')

if __name__ == "__main__":
    #getting training and testing data
    df = process_data(print_results=False)
    x_train_up  = df.x_train_over       # Oversampled training data
    y_train_up  = df.y_train_over       # Oversampled training target
    x_train_down = df.x_train_shrink    # Shrunk and oversampled training data
    y_train_down = df.y_train_shrink    # Shrunk and oversampled trainign target
    x_train = df.x_train                # Training data
    y_train = df.y_train                # Training targets
    x_test  = df.x_test                 # Test data
    y_test  = df.y_test                 # Test target

    tune_SVM()
