# Essensial libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

# Processing and metrics
from sklearn                  import metrics
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import cross_validate
from sklearn.model_selection  import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection  import StratifiedKFold
from sklearn.preprocessing    import LabelEncoder
from imblearn.over_sampling   import SMOTE
from imblearn.pipeline        import Pipeline


# Models
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC, LinearSVC

# Self made classes
from data_processing import process_data


def evaluate_model(model, x_train, x_test, y_train, y_test, rp_title, cm_title, rp_name, cm_name, save=True):
    """ Evaluate model and print accuracy, precision and recall. also plot the confusion matrix
    Arguments:
            model(class instance): instance of machine learning method from sklearn
            x_train(matrix): Design matrix for training data
            x_test(matrix): Design matriix for testing data
            y_train(array): target array for training data
            y_test(array): target array for testing data
            rp_title(string): recall percision AUC plot title
            cm_title(string):confusion matrix plot title
            rp_name(string): recall percision AUC plot filename
            cm_name(string): confusion matrix filename
            save (boolean, default = True): if True saves plot to visuals folder

    """
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred  = model.predict(x_test)

    # Run through metrics
    print("\nClassification Results on train")
    print("Accuracy: ",  metrics.accuracy_score(y_train, y_train_pred))
    print("Precision: ", metrics.precision_score(y_train, y_train_pred))
    print("Recall: ",    metrics.recall_score(y_train, y_train_pred))

    C_train = metrics.confusion_matrix(y_train, y_train_pred)
    print("Confusion matrix: (TN in top left)\n", C_train)

    print("\nClassification Results on test")
    print("Accuracy: ",  metrics.accuracy_score(y_test, y_test_pred))
    print("Precision: ", metrics.precision_score(y_test, y_test_pred))
    print("Recall: ",    metrics.recall_score(y_test, y_test_pred))

    C_test = metrics.confusion_matrix(y_test, y_test_pred)
    print("Confusion matrix: (TN in top left)\n", C_test)

    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
    fig.suptitle(cm_title, fontsize=16)

    sns.heatmap(C_train, ax=ax1, annot=True, cbar=False, cmap="Blues", square=True, fmt="d", annot_kws={"fontsize":12})
    sns.heatmap(C_test,  ax=ax2, annot=True, cbar=False, cmap="Blues", square=True, fmt="d", annot_kws={"fontsize":12})
    ax1.set_title("Train", fontsize=14)
    ax2.set_title("Test", fontsize=14)

    ax1.set_xlabel("Predicted label", fontsize=14)
    ax2.set_xlabel("Predicted label", fontsize=14)
    ax1.set_ylabel("True label", fontsize=14)

    ax1.tick_params(labelsize='12')
    ax2.tick_params(labelsize='12')

    # Save figures
    if save:
        plt.savefig("../visuals/" + cm_name + ".pdf")
    plt.show()

#    # Plot precision recall curve (only for test data)
#    y_prob = model.predict_proba(x_test)[:,1]
#    pres, rec, thresholds = metrics.precision_recall_curve(y_test, y_prob)
#
#    plt.style.use('seaborn-whitegrid')
#    no_skill = len(y_test[y_test==1]) / len(y_test)
#    plt.plot([0,1], [no_skill, no_skill], "--", label="No skill")
#    plt.plot(rec, pres, label = "model")
#    plt.xlabel("Recall", fontsize=14)
#    plt.ylabel("Precision", fontsize=14)
#    plt.title("Recall precision curve \n" + rp_title, fontsize=16)
#    plt.tick_params(labelsize='12')
#    plt.legend(fontsize=12)

    if save:
        plt.savefig("../visuals/" + rp_name + ".pdf")
    plt.show()

def bootstrap(model, x_train, x_test, y_train, y_test, n_iter=10, print_progress=True):
    """ Resamples training data and prins metrics
    model(class instance): instance of machine learning method from sklearn
    x_train(matrix): Design matrix for training data
    x_test(matrix): Design matriix for testing data
    y_train(array): target array for training data
    y_test(array): target array for testing data
    n_iter(int, default = 10): How many times to run bootstrap
    print_progress(boolean, default = True): if True prints progress for every loop
    """
    # Set up arrays for storage
    accuracy_scores = np.zeros(n_iter)
    recall_scores = np.zeros(n_iter)
    roc_auc_scores = np.zeros(n_iter)

    # Bootstrap
    for i in range(n_iter):
        # Resample training data
        idx = np.arange(len(y_train))
        idx = np.random.choice(idx, len(y_train))
        x_new = x_train[idx]
        y_new = y_train[idx]

        # Make fit and predict
        model.fit(x_new, y_new)
        y_pred = model.predict(x_test)

        # Store result
        accuracy_scores[i] = metrics.accuracy_score(y_test, y_pred)
        recall_scores[i] = metrics.recall_score(y_test, y_pred)
        roc_auc_scores[i] = metrics.roc_auc_score(y_test, y_pred)

        if print_progress:
            print("Iteration: %i/%i" %(i+1, n_iter))

    # Find average and variance
    print("\nBootstrapped results:")
    print("Accuracy: %.3f (%.3f)"%(np.mean(accuracy_scores), np.var(accuracy_scores)))
    print("Recall:   %.3f (%.3f)"%(np.mean(recall_scores), np.var(recall_scores)))
    print("ROC-AUC:  %.3f (%.3f)"%(np.mean(roc_auc_scores), np.var(roc_auc_scores)))


def tune_decision_tree():
    """
    tunes decision tree and plots training and test "error" for oversampled data
    """
    # Fetch model
    dt = DecisionTreeClassifier()

    # Evaluate without oversampling
    print("Decision tree, no oversampling, no tuning: ")
    cm_title = "Baseline confusion matrix: Decision tree"
    rp_title = "Decision tree: No oversampling, no tuning"
    cm_name = "cm_baseline_dt"
    rp_name = "rp_baseline_dt"
    evaluate_model(dt, x_train, x_test, y_train, y_test, rp_title, cm_title, rp_name, cm_name)

    # Implement oversampling
    print("\n\nDecision tree with oversampling, no tuning: ")
    cm_title = "Oversampled confusion matrix: Decision tree"
    rp_title = "Decision tree: SMOT oversampling, no tuning"
    cm_name = "cm_over_dt"
    rp_name = "rp_over_dt"
    evaluate_model(dt, x_train_up, x_test, y_train_up, y_test, rp_title, cm_title, rp_name, cm_name)

    # Perform random search--------------------------------------------------------------
    n_iter = 10
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
        #max_depth = max_depth_array[i],
        #criterion = criterion_array[i],
        #min_samples_split = min_samples_split_array[i],
        #min_samples_leaf = min_samples_leaf_array[i]
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
    exit()


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

def tune_SVM():
    """
    Finds best parameter values for SVM using grid search and evaluate model
    """
#C=  0.01 gamma=  0.00020691380811147902
#C=  0.01 gamma=  0.00069519279617756050.008858667904100823
#C=  0.008858667904100823 gamma=  0.00029763514416313193
    c_lst = [1,1.2,1.5,1.8,2,4]
    kernel_lst = ['linear','poly', 'sigmoid','rbf']
    gamma_lst = np.logspace(-1,0,10)

    for i in range(1):

            print (c_lst[i])
            model = SVC(kernel ='sigmoid', gamma = 0.36 ,C = 1.5, probability = True )
            evaluate_model(model, x_train_down, x_test, y_train_down, y_test, 'oversampled_CM_SVM', 'Tuned SVM with oversampling ', 'svm_rp_name', 'tuned_svm_cm_name')


def tune_random_forest():
    """
    Finds best parameter values for random forest using random search and prints best
    values
    """
    # Baseline with oversampling, no tuning
    rf = RandomForestClassifier()
    print("RandomForestClassifier, no oversampling, no tuning: ")
    cm_title = "Oversample confusion matrix: Random forest"
    rp_title = "Random forest: SMOTE oversampling, no tuning"
    cm_name = "rf_cm_over"
    rp_name = "rf_rp_over"
    #evaluate_model(rf, x_train_up, x_test, y_train_up, y_test, rp_title, cm_title, rp_name, cm_name)

    params ={
    "n_estimators":     list(np.arange(2,100,1)),
    "max_depth":        list(np.arange(2,10,1)),
    "min_samples_leaf": list(np.arange(2,10,1)),
    "min_samples_leaf": list(np.arange(2,10,1)),
    }

    # Set up model and fit to data
    search = RandomizedSearchCV(rf, params, n_iter=15)
    search.fit(x_train, y_train)
    print(search.best_params_)
    print(search.best_score_)

    rf_tune = RandomForestClassifier(search.best_params_)
    cm_title = "Oversample confusion matrix: Tuned random forest"
    rp_title = "Random forest: SMOTE oversampling, tuned"
    cm_name = "rf_cm_over_tune"
    rp_name = "rf_rp_over_tune"
    evaluate_model(search, x_train, x_test, y_train, y_test, rp_title, cm_title, rp_name, cm_name)



#getting training and testing data
df = process_data(print_results=False)
x_train_up  = df.x_train_over   # Oversampled training data
y_train_up  = df.y_train_over   # Oversampled training target
x_train_down = df.x_train_shrink
y_train_down = df.y_train_shrink
x_train = df.x_train
y_train = df.y_train
x_test  = df.x_test
y_test  = df.y_test


tune_SVM()
#
#
#tune_decision_tree()
#tune_random_forest()
