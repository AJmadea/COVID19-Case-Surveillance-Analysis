from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import Cleaning as c
import warnings


def find_targets(_data, x_columns, target_columns, k=5, include_targets=True):
    warnings.filterwarnings('ignore')
    all_col = x_columns
    all_col.extend(target_columns)
    _data = filter_columns(_data, list_of_columns=all_col)
    _data = c.drop_rows_with_nil_values(_data, all_col)
    _map = iterate_through_targets(_data, x_columns, target_columns, include_targets=include_targets, k=k)
    acc_df = create_acc_df(_map)
    acc_df.to_csv('data/private_data/private_data_target_cols.csv')
    graph_bar_acc(acc_df)


def iterate_through_targets(_data, x_columns, target_columns, include_targets=True, k=5):
    _map = {}

    dummies = pd.get_dummies(_data)
    print(x_columns)

    for col in target_columns:

        print(x_columns)
        print(target_columns)

        dummies = pd.get_dummies(_data)
        x_cols = []


        if include_targets:
            for _c in dummies.columns:
                if (_c != col + "_Yes") & (_c != col + "_No"):
                    x_cols.append(_c)
        else:
            for _c in dummies.columns:
                if (_c + "_Yes" not in target_columns) & (_c + "_No" not in target_columns):
                    x_cols.append(_c)

        col += '_Yes'

        print('xColumns:', x_cols)
        x = dummies[x_cols]
        print(dummies.columns)
        y = dummies[[col]]

        xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.7)

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
    _data = filter_columns(_data, columns)
    _map = iterate_through_k(_data, maxK, target)
    print(_map)
    acc_df = create_acc_df(_map)
    acc_df.to_csv('data/acc_df.csv')
    graph_line_acc(acc_df)


def graph_bar_acc(df):
    fig = px.bar(df, barmode='group', x='Index', y='Score', color='Score Type',
                 title='KNN Accuracy For Different Targets')
    fig.show()


def graph_line_acc(df):
    fig = px.line(df, x='Index', y='Score', color='Score Type', title='KNN Accuracy')
    fig.show()


def create_acc_df(_map):
    Ks = []
    scoreTypes = []
    scores = []
    for k in _map.keys():
        for i in range(0, 2):
            l = _map.get(k)
            Ks.append(k)
            scoreTypes.append('Jaccard Index' if i == 0 else 'F1 Score')
            scores.append(l[i])
    return pd.DataFrame(data={'Index': Ks,
                               'Score Type': scoreTypes,
                               'Score': scores}, columns=['Index', 'Score Type', 'Score'])


def filter_columns(_data, list_of_columns):
    for _c in _data.columns:
        if _c not in list_of_columns:
            _data.drop(_c, axis=1, inplace=True)
    return _data


def iterate_through_k(_data, maxK, target='death_yn'):
    _map = {}

    print('creating dummies for these columns:', _data.columns)
    dummies = pd.get_dummies(_data)

    xColumns = []

    for _c in dummies.columns:
        if (_c != target+"_Yes") & (_c != target+"_No"):
            xColumns.append(_c)

    target += '_Yes'

    print('xColumns:', xColumns)
    x = dummies[xColumns]
    print(dummies.columns)
    y = dummies[[target]]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.7)

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

        _map.__setitem__(k, [jaccard, f1])
        print(_map.get(k))
    return _map


