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
from sklearn.linear_model import RidgeClassifier


# Models
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.svm           import SVC

# Self made classes
from data_processing import process_data

#getting training and testing data
df = process_data(print_results=False)
x_train_up  = df.x_train_over   # Oversampled training data
y_train_up  = df.y_train_over   # Oversampled training target
x_train = df.x_train
y_train = df.y_train
x_test  = df.x_test
y_test  = df.y_test

def evaluate_model(model, x_train, x_test, y_train, y_test, rp_title, cm_title, rp_name, cm_name, save=True):
    """ Evaluate model by chosen metrics """
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

    # Plot precision recall curve (only for test data)
    y_prob = model.predict_proba(x_test)[:,1]
    pres, rec, thresholds = metrics.precision_recall_curve(y_test, y_prob)

    plt.style.use('seaborn-whitegrid')
    no_skill = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0,1], [no_skill, no_skill], "--", label="No skill")
    plt.plot(rec, pres, label = "model")
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Recall precision curve \n" + rp_title, fontsize=16)
    plt.tick_params(labelsize='12')
    plt.legend(fontsize=12)

    if save:
        plt.savefig("../visuals/" + rp_name + ".pdf")
    plt.show()

def bootstrap(model, x_train, x_test, y_train, y_test, n_iter=10, print_progress=True):
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

def evaluate_model_CV(model, n_folds):
    """ Evaluate given model using cross validation """
    # Define metrics
    scoring=['accuracy','average_precision','balanced_accuracy','f1','precision','recall','roc_auc']
    # Cross validate on training set
    cv = pd.DataFrame(cross_validate(model, x_train, y_train, scoring=scoring, cv=n_folds, shuffle=True))
    display(cv)

def tune_decision_tree():
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
    print("Starting parameter tuning: Support Vector Machine...")
    print("Making pipeline...")

    param_pipeline = {"classification__gamma": [0.1,10,100], 'classification__C': [0.1, 10,100], "classification__kernel": ['poly','rbf']}

    model = Pipeline([
            ('smt', SMOTE()),
            ('classification', SVC())])

    print("Performing grid search...")
    grid = GridSearchCV(model, param_pipeline)
    grid.fit(x_train, y_train)
    y_pred = grid.predict(x_test)
    evaluate_model(y_test, y_pred)

def tune_random_forest():
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
    search = RandomizedSearchCV(rf, params, n_iter=2)
    search.fit(x_train, y_train)
    print(search.best_params_)
    print(search.best_score_)

    cm_title = "Oversample confusion matrix: Tuned random forest"
    rp_title = "Random forest: SMOTE oversampling, tuned"
    cm_name = "rf_cm_over_tune"
    rp_name = "rf_rp_over_tune"
    evaluate_model(rf_tune, x_train_up, x_test, y_train_up, y_test, rp_title, cm_title, rp_name, cm_name)

tune_decision_tree(); exit()
#tune_random_forest(); exit()

def tune_logistic_regression():
    print("Starting parameter tuning: Logistic Regression...")

def kfold_with_smote(model):
    cv = StratifiedKFold(n_splits=5, random_state=42)
    smoter = SMOTE(random_state=42)

    # Arrays for storing scores
    accuracy_scores     = []
    precision_scores    = []
    recall_scores       = []
    roc_auc_scores      = []

    # Keeping track of progress
    current_iter = 1
    total_iter = 5

    print("Performing oversampled cross validation")
    for train_fold_index, val_fold_index in cv.split(x_train, y_train):
        # Fetch training and validation data
        x_train_fold, y_train_fold = x_train[train_fold_index], y_train[train_fold_index]
        x_val_fold, y_val_fold = x_train[val_fold_index], y_train[val_fold_index]

        # Oversample training data
        smoter = SMOTE(random_state=42)
        x_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(x_train_fold, y_train_fold)

        # Make prediction
        model.fit(x_train_fold_upsample, y_train_fold_upsample)
        y_pred = model.predict(x_val_fold)

        # Store metric scores
        accuracy_scores .append(metrics.accuracy_score(y_val_fold, y_pred))
        precision_scores.append(metrics.precision_score(y_val_fold, y_pred))
        recall_scores   .append(metrics.recall_score(y_val_fold, y_pred))
        roc_auc_scores  .append(metrics.roc_auc_score(y_val_fold, y_pred))

        # Print progress
        print(f"Iteration: {current_iter}/{total_iter}")
        current_iter+=1

    # Print final result
    print("avg Accuracy: ",     np.mean(accuracy_scores))
    print("avg Precision: ",    np.mean(precision_scores))
    print("avg Recall: ",       np.mean(recall_scores))
    print("avg ROC-AUC: ",      np.mean(roc_auc_scores))


#tune_SVM()
kernels = ['linear','poly','rbf','sigmoid']
cs = [0.001,0.1,1,100,1000]
gammas = [0.001,0.1,1,100,1000]

c_lst = np.logspace(-4,0,20)
plotting = []
#for i in range(10):
#    for k in range(10):
#        print ("values kernel, c, gamma", cs[i], gammas[k])
for k in range(20):
    for i in range(20):
        print (c_lst[i])
        model = SVC(kernel = 'sigmoid', C = c_lst[k] ,gamma = c_lst[i], probability = True )
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:,1]
        evaluate_model(y_test, y_pred)






#plt.plot(c_lst,plotting)
#plt.show()
#exit()
#
#print(" Making pipeline...")
#logreg = LogisticRegression()
#param = {"classification__penalty": ['l2'], 'classification__C': [0.1, 0.5, 1.0, 10, 100]}
#param = {"penalty": ['l2'], 'C': [0.1, 0.5, 1.0, 10, 100]}
##param = {'classification__max_leaf_nodes': list(range(2, 10)),
##        'classification__min_samples_split': [2, 3, 4]}
#logreg.fit(x_train, y_train)
#y_ped = logreg.predict(x_test)
#evaluate_model(y_test, y_pred)
#"""
#model = Pipeline([
#        ('smt', SMOTE()),
#        ('classification', LogisticRegression())])
#model = LogisticRegression()
#print("Performing grid search")
#grid = GridSearchCV(model, param)
#grid.fit(x_train, y_train)
#y_pred = grid.predict(x_test)
#evaluate_model(y_test, y_pred)
#"""
