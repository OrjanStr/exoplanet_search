# Essensial libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display

# Processing and metrics
from sklearn                  import metrics
from sklearn.model_selection  import train_test_split

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
        plt.savefig("../../visuals/" + cm_name + ".pdf")
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
        plt.savefig("../../visuals/" + rp_name + ".pdf")
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
