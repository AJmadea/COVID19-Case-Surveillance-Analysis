from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import plotly.express as px


def find_and_graph_K(_data):
    _map = iterate_through_k(_data)
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


def iterate_through_k(_data):
    _map = {}
    _data.drop(['race_ethnicity_combined', 'current_status'], axis=1, inplace=True)

    dummies = pd.get_dummies(_data[_data.columns[4:]])
    x = dummies[dummies.columns[:-2]]
    maxK = 10
    y = dummies[dummies.columns[-2:]]

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.7)
    yTest = yTest['death_yn_Yes'].to_numpy()
    for k in range(1, maxK):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(xTrain, yTrain)
        predicted = knn.predict(xTest)
        jaccard = jaccard_score(y_pred=predicted, y_true=yTest)
        f1 = f1_score(y_pred=predicted, y_true=yTest)
        _map.__setitem__(k, [jaccard, f1])
    return _map