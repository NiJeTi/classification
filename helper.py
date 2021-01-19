import os

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_data():
    dataset_path = os.path.join(os.path.dirname(__file__), 'Data.csv')
    dataset = pd.read_csv(dataset_path)

    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test


def calculate_metrics(y_predicted, y_test, name):
    cm = confusion_matrix(y_test, y_predicted)
    acc = accuracy_score(y_test, y_predicted)

    print(f'{name}:')
    print(cm)
    print(f"{(acc * 100):.2f} %")
