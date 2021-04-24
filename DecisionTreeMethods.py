from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import Cleaning as c
from sklearn.tree import plot_tree
from sklearn import tree
import matplotlib.pyplot as plt


def find_tree(_data, list_of_columns, target):
    c.drop_rows_with_nil_values(_data, in_these_columns=list_of_columns)
    _data = filter_columns(_data, list_of_columns)
    j, f = decision_tree(_data, list_of_columns, target)
    print('Jaccard: %2.4f\n F1: %2.4f' % (j,f))


def filter_columns(_data, list_of_columns):
    for _c in _data.columns:
        if _c not in list_of_columns:
            _data.drop(_c, axis=1, inplace=True)
    return _data


def decision_tree(_data, x_columns, target):
    print('creating dummies for these columns:', _data.columns)
    dummies = pd.get_dummies(_data)

    xColumns = []

    for _c in dummies.columns:
        if (_c != target + "_Yes") & (_c != target + "_No"):
            xColumns.append(_c)

    x = dummies[xColumns]
    print(dummies.columns)

    target += '_Yes'
    y = dummies[[target]]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.7)

    yTest = yTest[target].to_numpy()

    t = DecisionTreeClassifier(criterion='entropy')
    t.fit(xTrain, yTrain)
    yHat = t.predict(xTest)

    tree.plot_tree(t)
    plt.show()

    return jaccard_score(y_pred=yHat, y_true=yTest), f1_score(y_pred=yHat, y_true=yTest)