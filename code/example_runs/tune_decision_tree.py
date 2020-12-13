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

def random_search():
    max_depth_array = np.arange(2,10,1)
