from knn import KNearestNeighbor
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def test_model(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


dt = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    np.array(dt.data),
    np.array(dt.target),
    train_size=0.33
)
my_accuracy = test_model(KNearestNeighbor(k=3))
sklearn_accuracy = test_model(KNeighborsClassifier(n_neighbors=3))

print('My accuracy: {} \nsklearn accuracy: {}'.format(my_accuracy, sklearn_accuracy))