import pandas as pd
import plotly.express as px
from datetime import datetime as dt
import os
import Logistic as l
import KNNStuff as ks


def combine_dataframes(target):
    all_frames = []
    for f in os.listdir('data/acc_frames/{}'.format(target)):
        p = f.split('_')[4]
        current = pd.read_csv('data/acc_frames/{}/'.format(target) + f)
        current['Priority'] = p
        all_frames.append(current)
    concat_frame = pd.concat(all_frames)
    concat_frame.drop("Unnamed: 0", axis=1, inplace=True)
    concat_frame.rename({"Index": 'K'}, axis=1, inplace=True)
    concat_frame['Priority'] = concat_frame['Priority'].astype(int)

    jaccard = concat_frame[concat_frame['Score Type'] == 'Jaccard Index']
    jaccard.sort_values(by=['Priority', 'K'], inplace=True)
    print(jaccard.head(10))

    f1 = concat_frame[concat_frame['Score Type'] == 'F1 Score']
    f1.sort_values(by=['Priority', 'K'], inplace=True)

    jacc_fig = px.line(jaccard, x='K', y='Score', animation_frame='Priority', range_y=[.8, 1],
                       title='Jaccard Index Over K as More Columns are Introduced into KNN')
    jacc_fig.show()

    f1_fig = px.line(f1, x='K', y='Score', animation_frame='Priority', range_y=[.8, 1],
                     title='F1 Score Over K as More Columns are Introduced into KNN')
    f1_fig.show()


def create_x_y(_data, x_columns, target):
    _data = _data[x_columns]
    uniques = get_different_dummies_columns(_data[[target]])
    dummies = pd.get_dummies(_data, columns=x_columns)

    x_dummies = dummies.columns.tolist().copy()
    for col in dummies.columns:
        if col in uniques:
            x_dummies.remove(col)

    x = dummies[x_dummies]
    y = dummies[uniques[0]]
    return x, y


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


def get_different_dummies_columns(_data):
    columns = _data.columns
    c = []
    for col in columns:
        l = _data[col].unique()
        for e in l:
            if str(e) != "nan":
                c.append(str(col) + "_" + str(e))
    return c


def write_df_to_file(dataframe, filename):
    current = dt.today().isoformat().replace(':', '_').replace('-', '_')
    dataframe.to_csv(filename.format(current))


def get_cols():
    return ['race_ethnicity_combined', 'current_status', 'sex',
            'hosp_yn', 'icu_yn', 'death_yn', 'hc_work_yn', 'pna_yn', 'abxchest_yn',
            'acuterespdistress_yn', 'mechvent_yn', 'fever_yn', 'sfever_yn', 'chills_yn', 'myalgia_yn', 'runnose_yn',
            'sthroat_yn', 'cough_yn', 'sob_yn', 'nauseavomit_yn', 'headache_yn', 'abdom_yn', 'diarrhea_yn',
            'medcond_yn', 'res_state', 'age_group']


def update(data):
    cols = get_cols()
    cols.remove('res_state')

    for col in cols:
        current_columns = cols.copy()
        _m = l.use_rfecv(data, current_columns, col)
        target_dummies = get_different_dummies_columns(data[[col]])
        for target in target_dummies:
            ks.find_graph_k_dummies(data, dummies=_m[1], target=target, maxK=10, priority='nan')


def get_graphs(data):
    base = 'data/acc_frames/{}/'
    cols = get_cols()
    cols.remove('res_state')

    dummy_names = get_different_dummies_columns(data[cols])
    datas = []
    for e in dummy_names:
        d = base.format(e.replace('/', "_").replace(' ', '_'))
        temp = pd.read_csv(d + os.listdir(d)[0])
        temp['TARGET'] = e
        datas.append(temp)

    all_data = pd.concat(datas)
    f1_scores = all_data[all_data['Score Type'] == 'F1 Score']
    f1_scores = f1_scores[f1_scores['Score'] >= 0.5]
    jaccards = all_data[all_data['Score Type'] == 'Jaccard Index']
    jaccards = jaccards[jaccards['Score'] >= 0.5]

    f1_scores = f1_scores[f1_scores['Index'] == 5]
    jaccards = jaccards[jaccards['Index'] == 5]

    f1_scores.sort_values(by='Score', inplace=True, ascending=False)
    jaccards.sort_values(by='Score', inplace=True, ascending=False)

    f = px.bar(f1_scores, x='TARGET', y='Score', title='F1 Score For Each Target')
    f.show()
    j = px.bar(jaccards, x='TARGET', y='Score', title='Jaccard Index For Each Target')
    j.show()
    f1 = px.line(f1_scores, x='Index', y='Score', color='TARGET', title='F1 Scores')
    f1.show()
    jac_fig = px.line(jaccards, x='Index', y='Score', color='TARGET', title='Jaccard Index')
    jac_fig.show()
    all_data.to_csv('data/acc_frames/rfecv_4f_knn_1_to_10_april.csv')
