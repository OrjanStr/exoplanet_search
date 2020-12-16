# Essensial libraries
import numpy as np

# Processing and metrics
from sklearn                  import metrics

# Models
from sklearn.ensemble      import RandomForestClassifier

# Self made classes
import sys
sys.path.insert(0,"..")
from model_setup import evaluate_model
from data_processing import process_data

def tune_random_forest():
    """
    Finds best parameter values for random forest using random search and prints best
    values
    """
    # Baseline with oversampling, no tuning
    rf = RandomForestClassifier()
    print("RandomForestClassifier, oversampling, no tuning: ")
    cm_title = "Confusion matrix: Baseline random forest with oversampling"
    rp_title = "Random forest: No oversampling, no tuning"
    cm_name = "rf_cm"
    rp_name = "rf_rp"
    #evaluate_model(rf, x_train_up, x_test, y_train_up, y_test, rp_title, cm_title, rp_name, cm_name)

    # Using results from random search
    model = RandomForestClassifier(
    n_estimators = 41,
    max_depth = 2,
    min_samples_split = 4,
    min_samples_leaf = 7
    )
    print("RandomForestClassifier, Oversampling, tuned: ")
    cm_title = "Confusion matrix: Tuned random forest with oversampling"
    rp_title = "Random forest: Oversampled data, Tuned"
    cm_name = "rf_cm_tune"
    rp_name = "rf_rp_tune"
    evaluate_model(model, x_train_up, x_test, y_train_up, y_test, rp_title, cm_title, rp_name, cm_name)
    exit()

    # Random search --------------------------------------------------------------------------------
    n_iter = 20
    # Set up parameters
    params ={
    "n_estimators":     list(np.arange(2,100,1)),
    "max_depth":        list(np.arange(2,10,1)),
    "min_samples_leaf": list(np.arange(2,10,1)),
    "min_samples_split": list(np.arange(2,10,1)),
    }

    # Lists for storing used parameters
    n_estimators_array      = []
    max_depth_array         = []
    min_samples_split_array = []
    min_samples_leaf_array  = []

    # Lists for storing results
    scores = []
    predictions = []

    # Search
    print("Performing random seach...")
    for i in range(n_iter):
        # Fetch random parameters
        n_estimators_array      .append( np.random.choice( params["n_estimators"]) )
        max_depth_array         .append( np.random.choice( params["max_depth"]) )
        min_samples_split_array .append( np.random.choice( params["min_samples_split"]) )
        min_samples_leaf_array  .append( np.random.choice( params["min_samples_leaf"]) )

        # Set up model
        model = RandomForestClassifier(
        n_estimators        = n_estimators_array[i],
        max_depth           = max_depth_array[i],
        min_samples_split   = min_samples_split_array[i],
        min_samples_leaf    = min_samples_leaf_array[i]
        )

        # Make prediction
        model.fit(x_train_up, y_train_up)
        predictions.append( model.predict(x_test) )
        scores.append( metrics.recall_score(y_test, predictions[i]) )

        # Print progress
        print("Iteration %i/%i"%(i+1, n_iter))

    # Print best result
    idx = np.argmax(scores)
    print("n_estimators: ",      n_estimators_array      [idx])
    print("max_depth: ",         max_depth_array         [idx])
    print("min_samples_split: ", min_samples_split_array [idx])
    print("min_samples_leaf: ",  min_samples_leaf_array  [idx])
    print(metrics.confusion_matrix(y_test, predictions[idx]))

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

    tune_random_forest()
