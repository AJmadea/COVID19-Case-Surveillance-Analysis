import pandas as pd
import os
import glob


def get_private_data():
    return pd.read_csv('data/private_data/combined_csv.csv')


def combine_all_private_data():
    # Change the directory path to your data input directory
    os.chdir(r"C:\Users\Andrew\Desktop\covid19caseAnalysis\data\private_data")

    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    # Combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f, dtype='unicode') for f in all_filenames])

    # Export to csv
    combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')


def get_public_data(source='api'):
    if source == 'api': # API endpoint.  1000 rows
        src = 'https://data.cdc.gov/resource/vbim-akqf.csv'
    elif source == 'file':
        src = 'data/COVID-19_Case_surveillance_Public_Use_Data.csv'
    else:
        raise KeyError('source must be either "api" or "file"')
    _data = pd.read_csv(src)
    return _data

