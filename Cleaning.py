import pandas as pd
import plotly.express as px


def drop_useless(_data):
    candidate_columns = _data.columns[4:]
    for c in candidate_columns:
        print(c)
        _data = _data[(_data[c] != 'Missing') & (_data[c] != 'Unknown')]
    _data.to_csv('data/no_missing_no_unknown.csv')


def create_default_map(_n):
    _map = {}
    for i in range(0, _n):
        _map.__setitem__(i, 0)
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
        _map.__setitem__(s, t)

    # Creates a list of strings 'n values unknown or missing'
    columns = ['{} values unknown or missing'.format(i) for i in range(0, len(candidate_columns))]
    return pd.DataFrame(columns=['# Values Unknown or Missing', 'Freq'],
                        data={'# Values Unknown or Missing': columns,
                              'Freq': _map.values()
                              }
                        )


def get_value_count_graphs(_data):
    for each_col in _data.columns:
        fig = px.bar(_data[each_col].value_counts().sort_index(), title=each_col.upper())
        fig.show()


def get_data(source='api'):
    if source == 'api': # API endpoint.  1000 rows
        src = 'https://data.cdc.gov/resource/vbim-akqf.csv'
    elif source == 'file':
        src = 'data/COVID-19_Case_surveillance_Public_Use_Data.csv'
    else:
        raise KeyError('source must be either "api" or "file"')
    _data = pd.read_csv(src)
    return _data


def print_value_counts(_data):
    for c in _data.columns:
        print('Value Counts for: ', c)
        print(_data[c].value_counts())
        print('\n')