import pandas as pd
from scipy.stats import chi2_contingency
import GettingData as gd
import Cleaning as c
import plotly.express as px


def calculate_write_graph(columns):
    print('# Iterations:', len(columns)**2 - len(columns))  # O(n**2 - n)
    chi_df = find_chi2_from_dropped(columns)
    #dropped = drop_duplicates(chi_df)
    chi_df["Two Features"] = chi_df['Index'] + " " + chi_df["Columns"]
    chi_df.to_csv('data/p_value_from_dropped.csv')
    chi_df.rename({'CHI 2': 'P VALUE'}, axis=1, inplace=True)
    fig = px.bar(chi_df.sort_values(by='P VALUE'), x='P VALUE', y='Two Features', title='P Value from Chi 2')
    fig.show()


def drop_duplicates(data):
    print("Data before dropping...", data.shape)
    s = set({})  # a set of tuples
    for i in data.index:
        before = len(s)

        index = data.loc[i, 'Index']
        col = data.loc[i, 'Columns']

        if index < col:
            s.add((index, col))
        else:
            s.add((col, index))

        after = len(s)
        if after - before == 0:  # The last entry was a duplicate.  Get rid of it
            data.drop(index=i, axis=0, inplace=True)
    print('Data after dropping...', data.shape)
    return data


def find_chi2_from_dropped(columns):
    data = pd.read_csv('data/private_data/dropped_values_private_data.csv')
    chi2_numbers = []
    column_indices = []
    for _c in columns:
        if _c not in data.columns:
            columns.remove(_c)
            print(_c, 'not found in dataframe')

    indices = []
    cols = []
    s = set({})  # Ensures duplicates are not added.
    for a in columns:
        for b in columns:
            if a == b:  # Ensures that chi2 isn't calculated when a,b are the same column
                continue

            # Ensures duplicates are not added by checking length of a set of tuples.
            before = len(s)
            if a < b:
                s.add((a, b))
            else:
                s.add((b, a))
            after = len(s)
            print(after-before)
            if after-before == 0:
                continue

            # Adding to a list
            indices.append(a)
            cols.append(b)

            # Calculating the p value...
            print("Crosstable...")
            crosstab = pd.crosstab(index=data[a], columns=data[b])
            print("Calculating Chi 2")
            chi2 = chi2_contingency(crosstab)
            print(chi2[1])
            chi2_numbers.append(chi2[1])
            column_indices.append((a, b))

    chi_df = pd.DataFrame(data={'Index': indices,
                                'Columns': cols,
                                'CHI 2': chi2_numbers})
    print('Chi_df.shape:', chi_df.shape)
    return chi_df


# Please don't use this.  O(n**2 - n) * r
def find_chi2_corr(columns):
    data = gd.get_private_data()
    chi2_numbers = []
    column_indices = []

    for _c in columns:
        if _c not in data.columns:
            columns.remove(_c)
            print(_c, 'not found in dataframe')

    indices = []

    for a in columns:
        for b in columns:
            indices.append(a)
            print(a, b)
            temp = c.drop_rows_with_nil_values(data, [a, b])
            print("Crosstable...")
            crosstab = pd.crosstab(index=data[a], columns=data[b])
            print("Calculating Chi 2")
            chi2 = chi2_contingency(crosstab)
            print(chi2[1])
            chi2_numbers.append(chi2[1])
            column_indices.append((a, b))

    chi_df = pd.DataFrame(data={'Index': indices,
                                'Columns': columns * len(columns),
                                'CHI 2': chi2_numbers})

    chi_df.to_csv('data/acc_frames_chi2.csv')