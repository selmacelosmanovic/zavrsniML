import os
import numpy as np
import pandas as pd
from os.path import isfile, join
import csv
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def normalize_data(metric, value):
    min = 0
    max = 0
    if metric == 0:  # dit
        min = 0
        max = 100
    elif metric == 1:  # rfc
        min = 0
        max = 200
    elif metric == 2:  # lcom
        min = 0
        max = 2
    elif metric == 5:  # noch
        min = 0
        max = 100

    return (value - min) / (max - min)


def handle_data(data):
    i = 0
    for x in data.to_numpy():
        data.loc[i, 'dit'] = normalize_data(0, x[0])
        data.loc[i, 'rfc'] = normalize_data(1, x[1])
        data.loc[i, 'lcom'] = normalize_data(2, x[2])
        data.loc[i, 'noch'] = normalize_data(5, x[5])
        i = i + 1


def make_graph(data, metric):
    plt.bar(range(data[metric].size), data[metric].to_numpy(), color='#AED6DC')
    plt.grid(color='#d9d9d9', linestyle='-', linewidth=1, axis='y', alpha=0.7)
    plt.xlabel('KLASE', color='#737373')
    plt.ylabel('VRIJEDNOST METRIKE', color='#737373')
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['left'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['bottom'].set_color('none')

    plt.show()


float_formatter = "{:.3f}".format


def preprocess():
    path = input("Unesite putanju do direktorija sa projektom: ")

    folders = os.listdir(path)
    with open(join(path, 'dataFinal.csv'), 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dit", "rfc", "lcom", "mhf", "ahf", "noch"])

    i = 0
    for f in folders:
        if not isfile(join(path, f)):  # da dobijem sve foldere
            projectCSVPath = join(join(path, folders[i]), "class.csv")
            print(projectCSVPath)
            data = pd.read_csv(projectCSVPath)
            data.drop('file', axis='columns', inplace=True)
            data.drop('class', axis='columns', inplace=True)
            data.drop('type', axis='columns', inplace=True)
            handle_data(data)
            data.dropna(1, 'all', inplace=True)  # drop kolone u kojima je svaka vrijednost NaN
            si = SimpleImputer(strategy='mean')
            data.loc[:, 'dit'] = si.fit_transform(data.loc[:, 'dit'].values.reshape(-1, 1))
            data.loc[:, 'rfc'] = si.fit_transform(data.loc[:, 'rfc'].values.reshape(-1, 1))
            if data.columns.__contains__('lcom'):
                data.loc[:, 'lcom'] = si.fit_transform(data.loc[:, 'lcom'].values.reshape(-1, 1))
            if data.columns.__contains__('mhf'):
                data.loc[:, 'mhf'] = si.fit_transform(data.loc[:, 'mhf'].values.reshape(-1, 1))
            if data.columns.__contains__('ahf'):
                data.loc[:, 'ahf'] = si.fit_transform(data.loc[:, 'ahf'].values.reshape(-1, 1))
            data.loc[:, 'noch'] = si.fit_transform(data.loc[:, 'noch'].values.reshape(-1, 1))
            data = data.mean(0)

            with open(join(path, 'dataFinal.csv'), 'a') as file1:
                np.set_printoptions(formatter={'float_kind': float_formatter})
                file1.write(','.join(map(str, data.to_numpy())))
                file1.write("\n")
            i = i + 1


preprocess()
