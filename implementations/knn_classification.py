from sklearn.neighbors import KNeighborsClassifier

from helper import get_data, preprocess_data, calculate_metrics

name = 'K-nearest Neighbors'

if __name__ == '__main__':
    x, y = get_data()
    x_train, x_test, y_train, y_test = preprocess_data(x, y)

    classifier = KNeighborsClassifier()
    classifier.fit(x_train, y_train)

    y_predicted = classifier.predict(x_test)

    calculate_metrics(y_predicted, y_test, name)