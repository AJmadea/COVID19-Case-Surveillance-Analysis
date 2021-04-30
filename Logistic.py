from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import pandas as pd
import ModelMethods as mm


def use_rfe(_data, x_columns, target):
    x, y = mm.create_x_y(_data, x_columns, target)

    print(x.shape)
    print(x.columns)

    lm = LogisticRegression(n_jobs=-1, verbose=True, max_iter=10_000)
    print("Creating RFE")
    rfe = RFE(estimator=lm, verbose=True)
    print("Fitting training data")
    rfe.fit(x, y)
    print(rfe.ranking_)
    m = create_map_ranking(rfe.ranking_, x.columns)

    with open('data/rfe_{}_columns_results.txt'.format(x.shape[1]), 'w') as f:
        f.write('Columns analyzed:' + " " + x.columns.__str__())
        f.write('\nMap Generated from analysis:\n')
        for i in sorted(m.keys()):
            f.write('{} : {}\n'.format(i, m.get(i).__str__()))
    return m


def use_rfecv(_data, x_columns, target):
    x, y = mm.create_x_y(_data, x_columns, target)
    print(x.shape)
    print(x.columns)

    lm = LogisticRegression(n_jobs=-1, verbose=True, max_iter=10_000)
    print("Creating RFECV")
    rfe = RFECV(estimator=lm, verbose=True, cv=4)
    print("Fitting training data")
    rfe.fit(x, y)
    print(rfe.ranking_)
    m = create_map_ranking(rfe.ranking_, x.columns)

    with open('data/rfecv_{}_columns_results.txt'.format(x.shape[1]), 'w') as f:
        f.write('Columns analyzed:' + " " + x.columns.__str__())
        f.write('\nMap Generated from analysis:\n')
        for i in sorted(m.keys()):
            f.write('{} : {}\n'.format(i, m.get(i).__str__()))
    return m


def create_map_ranking(rankings, columns):
    _map = {}
    for i, c in enumerate(rankings):
        if c in _map.keys():
            t = _map.get(c)
            t.append(columns[i])
            _map[c] = t
        else:
            _map[c] = [columns[i]]
    return _map


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
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=.7)

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
