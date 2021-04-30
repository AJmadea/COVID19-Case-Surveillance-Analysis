from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import Cleaning as c
import warnings
from datetime import datetime as dt
import ModelMethods as mm


def find_targets(_data, x_columns, target_columns, include_targets, k=5):
    warnings.filterwarnings('ignore')

    all_col = x_columns.copy()
    all_col.extend(target_columns.copy())
    _data = mm.filter_columns(_data, list_of_columns=all_col)
    _data = c.drop_rows_with_nil_values(_data, all_col)
    _map = iterate_through_targets(_data, x_columns, target_columns, include_targets=include_targets, k=k)
    acc_df = mm.create_acc_df(_map)

    mm.write_df_to_file(acc_df, 'data/acc_frames/private_data_target_cols_{}.csv')

    mm.graph_bar_acc(acc_df)


def iterate_unique_graph_write(_data, target, k):
    _data = c.drop_useless(_data)

    _map = iterate_through_unique(_data, target, k)
    acc_df = mm.create_acc_df(_map)

    mm.write_df_to_file(acc_df, 'data/acc_frames/acc_uniques_{}.csv')

    mm.graph_bar_acc(acc_df)


def iterate_through_unique(_data, target, k=5):
    _map = {}

    dummies = pd.get_dummies(_data)
    uniques = mm.get_different_dummies_columns(_data[[target]])

    for u in uniques:
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

        print("Creating KNN & k=", k)
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

        print('Currently fitting...')
        knn.fit(xTrain, yTrain)

        print("Predicting...")
        predicted = knn.predict(xTest)

        print("Calculating Accuracy")
        jaccard = jaccard_score(y_pred=predicted, y_true=yTest)
        f1 = f1_score(y_pred=predicted, y_true=yTest)

        _map[u] = [jaccard, f1]
        print(_map.get(u))
    return _map


def iterate_through_targets(_data, x_columns, target_columns, include_targets, k=5):
    _map = {}
    print('include other targets:', include_targets)
    print('x_columns', x_columns)
    print('targets from params:', target_columns)
    x_col_data = _data[x_columns]
    for col in target_columns:
        temp_data = None
        if include_targets:
            temp_data = pd.concat([x_col_data, _data[target_columns]], axis=1)
        else:
            temp_data = pd.concat([x_col_data, _data[col]], axis=1)

        dummies = pd.get_dummies(temp_data, columns=x_columns)
        print(dummies.columns)

        x_cols = dummies.columns.tolist().copy()
        #x_cols.remove(col)

        col += '_No'

        print('x_cols:', x_cols)
        x = dummies[x_cols]
        y = dummies[[col]]
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=.7)

        yTest = yTest[col].to_numpy()
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        print('Current Target: ', col)
        print('Currently fitting...')
        knn.fit(xTrain, yTrain)

        print("Predicting")
        predicted = knn.predict(xTest)

        print("Calculating Accuracy")
        jaccard = jaccard_score(y_pred=predicted, y_true=yTest)
        f1 = f1_score(y_pred=predicted, y_true=yTest)

        _map.__setitem__(col, [jaccard, f1])
        print(_map.get(col))
    return _map


def find_and_graph_K(_data, columns, maxK, target):
    warnings.filterwarnings('ignore')
    _data = c.drop_rows_with_nil_values(_data, columns)
    _data = mm.filter_columns(_data, columns)
    _map = iterate_through_k(_data, maxK, target)
    print(_map)
    acc_df = mm.create_acc_df(_map)
    acc_df.to_csv('data/acc_df.csv')
    mm.graph_line_acc(acc_df)


def find_graph_k_dummies(_data, dummies, maxK, target, priority):

    _map = {}

    print(dummies)
    dummy_data = pd.get_dummies(_data)
    x = dummy_data[dummies]
    y = dummy_data[[target]]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=.7)
    yTest = yTest[target].to_numpy()
    for k in range(1, maxK+1):
        print('k=',k)
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        print('Fitting...')
        knn.fit(xTrain, yTrain)
        print("Predicting...")
        predict = knn.predict(xTest)
        jaccard = jaccard_score(y_true=yTest, y_pred=predict)
        f1 = f1_score(y_true=yTest, y_pred=predict)
        _map[k] = [jaccard, f1]
        print(_map[k])
    print(_map)
    acc_df = mm.create_acc_df(_map)
    priority = str(priority)
    base = 'data/acc_frames/dummy_acc_df_priority_{}'.format(priority)
    mm.write_df_to_file(acc_df, base + '_{}.csv')
    mm.graph_line_acc(acc_df)



def iterate_through_k(_data, maxK, target='death_yn'):
    _map = {}


    print('creating dummies for these columns:', _data.columns)
    dummies = pd.get_dummies(_data)

    xColumns = []

    for _c in dummies.columns:
        if (_c != target+"_Yes") & (_c != target+"_No"):
            xColumns.append(_c)

    target += '_No'

    print('xColumns:', xColumns)
    x = dummies[xColumns]
    print(dummies.columns)
    y = dummies[[target]]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=.7)

    yTest = yTest[target].to_numpy()

    print(yTest)
    for k in range(1, maxK):
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        print('K=', k)
        print('Currently fitting...')
        knn.fit(xTrain, yTrain)

        print("Predicting")
        predicted = knn.predict(xTest)

        print("Calculating Accuracy")
        jaccard = jaccard_score(y_pred=predicted, y_true=yTest)
        f1 = f1_score(y_pred=predicted, y_true=yTest)

        _map[k] = [jaccard, f1]
        print(_map.get(k))
    return _map