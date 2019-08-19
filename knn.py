import numpy as np
from scipy.spatial.distance import cdist


class KNearestNeighbor:
    def __init__(self, k, dist_metric='euclidean'):
        """
        :type dist_metric: str. The distance metric to use.
        If a string, the distance function can be ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
        ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘mahalanobis’, ‘matching’,
        ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’,
        ‘wminkowski’, ‘yule’
        :k: int, count of neighbors
        initiate data and params for finding neighbors
        """
        self.base_data = np.array([])
        self.base_target = np.array([])
        self.k = k
        self.dist_metric = dist_metric

    def fit(self, x_train, y_train, expand=False):
        """
        :param y_train: 1d np.array
        :param expand: bool
        :param x_train: 2d np.array
        :fit model: write X_train in buffer, for finding neighbors in predicting:
        """
        self.base_data = x_train if not expand else self.base_data + x_train
        self.base_target = y_train if not expand else self.base_target + y_train

    def predict(self, x_test):
        """
        :rtype: np.array
        :type x_test: 2d np.array
        :return: prediction by x_test
        """

        # Создаем матрицу, содержащую расстояния между каждой парой из двух наборов
        # (того, на котором обучались и того, который предсказываем)
        # сортируем, отбираем индексы к строк с наименьшим расстоянием по выбранной метрике
        idx = np.argsort(cdist(x_test, self.base_data, metric=self.dist_metric))[:, :self.k]
        #
        y_pred = [np.bincount(self.base_target[i]).argmax() for i in idx]
        return y_pred
