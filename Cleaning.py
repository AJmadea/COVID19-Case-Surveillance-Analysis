import pandas as pd
import plotly.express as px


def drop_rows_with_nil_values(_data, in_these_columns):
    # Verify that all columns in in_these_columns exists as a column
    print("Validating that these columns exist in the dataframe...")
    for c in in_these_columns:
        if c not in _data.columns:
            print(c, 'is not in the dataframe!')
            in_these_columns.remove(c)

    intersection = set(_data.columns).intersection(set(in_these_columns))
    for c in intersection:
        print('Currently working on:', c)
        _data = _data[(_data[c] != 'Missing') & (_data[c] != 'Unknown')]
    print('After Dropping the nil values:', _data.shape)
    return _data


def drop_useless(_data):
    _data.dropna(inplace=True)
    for c in _data.columns:
        print('Currently working on: ', c)
        _data = _data[(_data[c] != 'Missing') & (_data[c] != 'Unknown')]

    #_data.to_csv('data/no_missing_no_unknown.csv')
    return _data


def create_default_map(_n):
    _map = {}
    for i in range(0, _n):
        _map[i] = 0
    return _map


def create_map_of_unknown_or_missing(_data):
    candidate_columns = _data.columns[4:]

    print("Initializing the map...")
    _map = create_default_map(len(candidate_columns))

    print('Creating the Frequency Table...')
    print(_data.index)
    for i in _data.index:
        a = _data.loc[i, candidate_columns]
        listA = list(a)
        s = listA.count('Unknown') + listA.count('Missing')
        print(float(i)/_data.index.max() * 100)

        t = _map.get(s)
        t += 1
        _map[s] = t

    # Creates a list of strings 'n values unknown or missing'
    columns = ['{} values unknown or missing'.format(i) for i in range(0, len(candidate_columns))]
    return pd.DataFrame(columns=['# Values Unknown or Missing', 'Freq'],
                        data={'# Values Unknown or Missing': columns,
                              'Freq': _map.values()
                              }
                        )


def get_value_count_graphs(_data):
    for each_col in _data.columns:
        print("Currently in", each_col)
        fig = px.bar(_data[each_col].value_counts().sort_index(), title=each_col.upper())
        fig.show()


def print_value_counts(_data):
    for c in _data.columns:
        print('Value Counts for: ', c)
        print(_data[c].value_counts())
        print('\n')