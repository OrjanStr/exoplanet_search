# Import models
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

# Import things
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

# Other essensials
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt

df     = pd.read_csv("../data/heart.csv")
input  = df.copy()
target = input.pop('target')

test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(input, target, test_size=test_size, random_state=42)
#cols = x_train.columns.to_list()

def model_eval(model, name):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:,1]
    acc_score = model.score(x_test, y_test)

    fpr, trp, thresholds = metrics.roc_curve(y_test, y_prob)
    plt.plot(fpr, trp, label=name)



# Logistic Regression
print("Logistic Regression:")
LR = LogisticRegression()
model_eval(LR, 'Logistic Reg')

# Decision Tree
print("Decision Tree:")
DT = DecisionTreeClassifier()
model_eval(DT, 'Decision Tree')

# Random Forest
print("Random Forest:")
RF = RandomForestClassifier()
model_eval(RF, 'Random Forest')

# XG Boost
print("XG Boost:")
XGB = XGBClassifier()
model_eval(XGB, 'XG boost')

plt.plot([0,1], [0,1], '--')
plt.legend()
plt.show()
