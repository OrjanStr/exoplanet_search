# Essensial libraries
import numpy as np
import matplotlib.pyplot as plt

# Processing and metrics
from sklearn                  import metrics
from sklearn.model_selection  import train_test_split

# Models
from sklearn.tree          import DecisionTreeClassifier

# Self made classes
import sys
sys.path.insert(0,"..")
from model_setup import evaluate_model
from data_processing import process_data

def tune_decision_tree():
    """ Tunes decision tree and creates a plot of errors showing overfitting"""
    # Study model result from search:
    model = DecisionTreeClassifier(
    max_depth = 5,
    criterion = 'entropy',
    min_samples_split = 7,
    min_samples_leaf = 3
    )

    print("\n\nDecision tree with oversampling, tuned: ")
    cm_title = "Confusion matrix: Tuned decision tree"
    rp_title = "Decision tree: No oversampling, tuned"
    cm_name = "cm_tuned_dt"
    rp_name = "rp_tuned_dt"
    #evaluate_model(model, x_train, x_test, y_train, y_test, rp_title, cm_title, rp_name, cm_name)

    # Fetch model
    dt = DecisionTreeClassifier()

    # Evaluate without oversampling
    print("Decision tree, no oversampling, no tuning: ")
    cm_title = "Baseline confusion matrix: Decision tree"
    rp_title = "Decision tree: No oversampling, no tuning"
    cm_name = "cm_baseline_dt"
    rp_name = "rp_baseline_dt"
    #evaluate_model(dt, x_train, x_test, y_train, y_test, rp_title, cm_title, rp_name, cm_name)

    # Implement oversampling
    print("\n\nDecision tree with oversampling, no tuning: ")
    cm_title = "Oversampled confusion matrix: Decision tree"
    rp_title = "Decision tree: SMOT oversampling, no tuning"
    cm_name = "cm_over_dt"
    rp_name = "rp_over_dt"
    evaluate_model(dt, x_train_up, x_test, y_train_up, y_test, rp_title, cm_title, rp_name, cm_name)
    exit()

    # Perform random search--------------------------------------------------------------
    n_iter = 25
    # Set up parameters
    max_depth = range(2,10)
    criterion = ['gini', 'entropy']
    min_samples_split = range(2,10)
    min_samples_leaf = range(2,5)

    # Arrays for storing used parameters
    max_depth_array           = []
    criterion_array           = []
    min_samples_split_array   = []
    min_samples_leaf_array    = []

    # Storing result
    accuracy_scores = []
    predictions = []

    # Search
    print("Performing random search")
    for i in range(n_iter):
        # Fetch random parameters
        max_depth_array.append(np.random.choice(max_depth))
        criterion_array.append(np.random.choice(criterion))
        min_samples_split_array.append(np.random.choice(min_samples_split))
        min_samples_leaf_array.append(np.random.choice(min_samples_leaf))

        # Set up model
        model = DecisionTreeClassifier(
        max_depth = max_depth_array[i],
        criterion = criterion_array[i],
        min_samples_split = min_samples_split_array[i],
        min_samples_leaf = min_samples_leaf_array[i]
        )

        # Make prediction
        model.fit(x_train_up, y_train_up)
        y_pred = model.predict(x_test)
        predictions.append(y_pred)
        accuracy_scores.append(metrics.recall_score(y_test, y_pred))

        # Print progress
        print("Iteration: %i/%i" %(i+1, n_iter))

    idx = np.argmax(accuracy_scores)
    print("max_depth: ", max_depth_array[i])
    print("criterion: ", criterion_array[i])
    print("min_samples_split: ", min_samples_split_array[i])
    print("min_samples_leaf: ", min_samples_leaf_array[i])
    print(metrics.confusion_matrix(y_test, predictions[idx]))


    # Bootstrap
    print("\n\nBootstrap resampling on oversampled model")
    #bootstrap(dt, x_train, x_test, y_train, y_test, n_iter = 10);

    # Study how max depth affects overfitting --------------------------------------------
    print("Starting parameter evaluation Decision Tree..\n")

    max_depth_array     = np.arange(3, 13, 1)
    train_score         = np.zeros(len(max_depth_array))
    valid_score         = np.zeros(len(max_depth_array))

    # Split training set into training and validation
    x_train_2, x_validate, y_train_2, y_validate = train_test_split(x_train_up, y_train_up, test_size=0.2)

    # Loop through depths
    for i, max_depth in enumerate(max_depth_array):
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(x_train_2, y_train_2)
        train_pred = model.predict(x_train_2)
        valid_pred = model.predict(x_validate)

        train_score[i] = metrics.recall_score(y_train_2, train_pred)
        valid_score[i] = metrics.recall_score(y_validate, valid_pred)

        # Print progress
        print(f"Current iteration: {i+1}/{len(max_depth_array)}")

    # Plot
    plt.style.use('seaborn-whitegrid')
    plt.plot(max_depth_array, train_score, label = "Train score")
    plt.plot(max_depth_array, valid_score, label = "Validation score")
    plt.ylabel("Recall score", fontsize=14)
    plt.xlabel("Max depth", fontsize=14)
    plt.title("Overfitting decision tree on oversampled train and validation", fontsize=16)
    plt.legend()
    plt.savefig("../visuals/overfitting_dt_depth.pdf")
    plt.show()

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

    tune_decision_tree()
