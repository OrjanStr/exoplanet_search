# Essensial libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

# Processing and metrics
from sklearn                  import metrics
from sklearn.model_selection  import train_test_split
from sklearn.model_selection  import cross_validate
from sklearn.model_selection  import GridSearchCV
from sklearn.model_selection  import StratifiedKFold
from sklearn.preprocessing    import LabelEncoder
from imblearn.over_sampling   import SMOTE
from imblearn.pipeline        import Pipeline


# Models
from sklearn.linear_model  import LogisticRegression
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from xgboost               import XGBClassifier
from sklearn.svm           import SVC


# Self made classes
from data_processing import process_data

df = process_data(print_results=False)
x_train = df.x_train
y_train = df.y_train
x_test  = df.x_test
y_test  = df.y_test

def evaluate_model(y_test, y_pred):
    """ Evaluate model by conventional train test split """

    # Run through metrics
    print("\nClassification Results")
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    print("Precision: ", metrics.precision_score(y_test, y_pred))
    print("Recall: ", metrics.recall_score(y_test, y_pred))
    print("ROC-AUC: ", metrics.roc_auc_score(y_test, y_pred))

    # Confusion Matrix
    print("Confusion matrix: (TN in top left)")
    print(metrics.confusion_matrix(y_test, y_pred))
    """
    # ROC-plot
    fpr, trp, thresholds = metrics.roc_curve(y_test, y_prob)
    plt.plot(fpr, trp)
    plt.show()
    """
#evaluate_model_TS();
def evaluate_model_CV(model, n_folds):
    """ Evaluate given model using cross validation """
    # Define metrics
    scoring=['accuracy','average_precision','balanced_accuracy','f1','precision','recall','roc_auc']
    # Cross validate on training set
    cv = pd.DataFrame(cross_validate(model, x_train, y_train, scoring=scoring, cv=n_folds, shuffle=True))
    display(cv)

def tune_decision_tree():
    print("Starting parameter evaluation Decision Tree..\n")

    # Study how max depth affects overfitting --------------------------------------------

    max_depth_array     = np.arange(1, 10, 1)
    train_score         = np.zeros(len(max_depth_array))
    valid_score         = np.zeros(len(max_depth_array))

    # Split training set into training and validation
    x_train_2, x_validate, y_train_2, y_validate = train_test_split(x_train, y_train, test_size=0.2)

    # Loop through depths
    for i, max_depth in enumerate(max_depth_array):
        model = DecisionTreeClassifier(max_depth=max_depth)
        model.fit(x_train, y_train)
        train_pred = model.predict(x_train)
        valid_pred = model.predict(x_test)

        train_score[i] = metrics.recall_score(y_train, train_pred)
        valid_score[i] = metrics.recall_score(y_test, valid_pred)

        # Print progress
        print(f"Current iteration: {i+1}/{len(max_depth_array)}")

    # Plot
    plt.plot(max_depth_array, train_score, label = "Train score")
    plt.plot(max_depth_array, valid_score, label = "Validation score")
    plt.ylabel("Recall score")
    plt.xlabel("max depth")
    plt.title("Overfitting for DecisionTreeClassifier on train and validation")
    plt.legend(); plt.show()

    # Grid search CV ---------------------------------------------------------------------
    params = {
    "max_depth": [2, 3, 4, 5, 6, 7],
    "criterion": ["gini", "entropy"],
    }

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

smoter = SMOTE()
#kfold_with_smote(model)
param = {'max_depth': [2,4,5]}
x_train, y_train = smoter.fit_resample(x_train, y_train)
grid = RandomForestClassifier(n_estimators = 100, max_depth=10)
print("Performing grid search")
#grid = GridSearchCV(model, param, verbose=True)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
evaluate_model(y_test, y_pred)
exit()

print(" Making pipeline...")
logreg = LogisticRegression()
param = {"classification__penalty": ['l2'], 'classification__C': [0.1, 0.5, 1.0, 10, 100]}
param = {"penalty": ['l2'], 'C': [0.1, 0.5, 1.0, 10, 100]}
#param = {'classification__max_leaf_nodes': list(range(2, 10)),
#        'classification__min_samples_split': [2, 3, 4]}
logreg.fit(x_train, y_train)
y_ped = logred.predict(x_test)
evaluate_model(y_test, y_pred)
"""
model = Pipeline([
        ('smt', SMOTE()),
        ('classification', LogisticRegression())])
model = LogisticRegression()
print("Performing grid search")
grid = GridSearchCV(model, param)
grid.fit(x_train, y_train)
y_pred = grid.predict(x_test)
evaluate_model(y_test, y_pred)
"""
