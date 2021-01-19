from sklearn.linear_model import LogisticRegression

from helper import get_data, calculate_metrics

name = 'Logistic Regression'

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    y_predicted = classifier.predict(x_test)

    calculate_metrics(y_predicted, y_test, name)
