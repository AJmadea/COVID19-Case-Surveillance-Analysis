from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import pandas as pd

import ModelMethods as mm
# get rid of certain columns


def iterate_through_dummies(_data, x_columns, target):
    _map = {}
    dummies = pd.get_dummies(_data, columns=x_columns)
    uniques = mm.get_different_dummies_columns(_data[[target]])

    for u in uniques:
        print('Uniques:', uniques)
        x_dummies = dummies.columns.tolist()
        for col in dummies.columns:
            if col in uniques:
                x_dummies.remove(col)

        print('x_dummies: ', x_dummies)
        x = dummies[x_dummies]
        y = dummies[u]

        print("Current Dummy Target", u)
        print('Splitting...')
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.7)

        print("Creating LogReg")
        lm = LogisticRegression(n_jobs=-1)

        print('Currently fitting...')
        lm.fit(xTrain, yTrain)

        print("Predicting...")
        predicted = lm.predict(xTest)

        print("Calculating Accuracy")
        _map[u] = log_loss(y_true=yTest, y_pred=predicted)
        print(_map.get(u))
    return pd.DataFrame(data={'Index': _map.keys(),
                        'Score': _map.values()})


def create_evaluate_model(data, x_columns, target, write_accuracy, make_graph):
    acc_df = iterate_through_dummies(data, x_columns, target)

    if write_accuracy:
        mm.write_df_to_file(acc_df, filename='log_reg_acc_{}.csv')

    if make_graph:
        mm.graph_bar_acc(acc_df)
