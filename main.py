import os
import numpy as np
import pandas as pd
from os.path import isfile, join
import csv
import matplotlib


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
        print(x)
        data.loc[i, 'dit'] = normalize_data(0, x[0])
        data.loc[i, 'rfc'] = normalize_data(1, x[1])
        data.loc[i, 'lcom'] = normalize_data(2, x[2])
        data.loc[i, 'noch'] = normalize_data(5, x[5])
        i = i + 1


def print_hi():
    path = input("Enter your value: ")

    folders = os.listdir(path)
    with open(join(path, 'dataset.csv'), 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["dit", "rfc", "lcom", "mhf", "ahf", "noch", "points"])

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
            print(data)
            i = i + 1


print_hi()
