from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px
import Cleaning as c


def find_and_graph_K(_data, columns, maxK):
    _data = c.drop_rows_with_nil_values(_data, columns)
    _data = filter_columns(_data, columns)
    _map = iterate_through_k(_data, maxK)
    print(_map)
    acc_df = create_acc_df(_map)
    graph_acc(acc_df)


def graph_acc(df):
    fig = px.line(df, x='K', y='Score', color='Score Type', title='KNN Accuracy')
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
    return pd.DataFrame(data={'K': Ks,
                               'Score Type': scoreTypes,
                               'Score': scores}, columns=['K', 'Score Type', 'Score'])


def filter_columns(_data, list_of_columns):
    for _c in _data.columns:
        if _c not in list_of_columns:
            _data.drop(_c, axis=1, inplace=True)
    return _data


def iterate_through_k(_data, maxK):
    _map = {}

    print('creating dummies for these columns:', _data.columns)
    dummies = pd.get_dummies(_data)

    xColumns = []

    for c in dummies.columns:
        if (c != 'death_yn_Yes') & (c != 'death_yn_No'):
            xColumns.append(c)

    print('xColumns:', xColumns)
    x = dummies[xColumns]
    print(dummies.columns)
    y = dummies[['death_yn_Yes']]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.7, random_state=100)

    yTest = yTest['death_yn_Yes'].to_numpy()

    print(yTest)
    for k in range(1, maxK):
        knn = KNeighborsClassifier(n_neighbors=k)
        print('K=', k)
        print('Currently fitting...')
        knn.fit(xTrain, yTrain)

        print("Predicting")
        predicted = knn.predict(xTest)

        print("Calculating Accuracy")
        jaccard = jaccard_score(y_pred=predicted, y_true=yTest)
        f1 = f1_score(y_pred=predicted, y_true=yTest)

        _map.__setitem__(k, [jaccard, f1])
    return _map