from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import numpy as np
import openml

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def load_mnist() -> Dataset:
    """Loads the MNIST dataset using OpenML https://www.openml.org/d/554"""
    mnist_openml = openml.datasets.get_dataset(554)
    Xy, _, _, _ = mnist_openml.get_data(dataset_format="array")
    # 資料（特徵，屬性）
    X = Xy[:, :-1]  # the last column contains labels
    # 資料的標籤
    y = Xy[:, -1]
    x_train, y_train = X[:60000], y[:60000]
    x_test, y_test = X[60000:], y[60000:]
    return (x_train, y_train), (x_test, y_test)


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions),
            np.array_split(y, num_partitions)))


def set_initial_params(model: LogisticRegression):

    n_classes = 10  # MNIST has 10 classes
    n_features = 784  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes, ))


#Returns the paramters of a sklearn LogisticRegression model.
def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    if model.fit_intercept:
        params = (model.coef_, model.intercept_)
    else:
        params = (model.coef_, )
    return params


# Sets the parameters(參數) of a sklean LogisticRegression model.
def set_model_params(model: LogisticRegression,
                     params: LogRegParams) -> LogisticRegression:
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def train(model: LogisticRegression):

    # downloads the training and test data
    (X_train, y_train), (X_test, y_test) = load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = partition(X_train, y_train, 10)[partition_id]

    model.fit(X_train, y_train)
    ac = model.score(X_test, y_test)
    parameters = get_model_parameters(model)
    model = set_model_params(model, parameters)

    return ac, model
