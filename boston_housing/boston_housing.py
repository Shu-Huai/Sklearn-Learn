import joblib
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn

from config import *
from utils import *


def load_data(test_size: float = 0.2) -> tuple[DataFrame, DataFrame, Series, Series]:
    data: DataFrame = pd.read_csv(r".\datasets\boston.csv")
    y: Series = data['medv']
    x: DataFrame = data.drop(['medv'], axis=1)
    return train_test_split(x, y, test_size=test_size)


def build_model() -> LinearRegression:
    return LinearRegression()


def fit(model: LinearRegression, x_train: DataFrame, y_train: Series) -> None:
    model.fit(x_train, y_train)


def score(test_value: ndarray, predict_value: ndarray) -> float:
    return r2_score(test_value, predict_value)


def predict(model: LinearRegression, x_test: DataFrame) -> ndarray:
    return model.predict(x_test)


def show_accuracy(test_value: ndarray, predict_value: ndarray, save: bool = True) -> None:
    plot(save, os.path.join(os.path.dirname(__file__), plot_path, 'accuracy.png') if save else None,
         test_value=test_value,
         predict_value=predict_value,
         difference=test_value - predict_value)


def print_parameter(model: LinearRegression, data: DataFrame) -> None:
    print("系数：" + str(model.coef_))
    print("截距" + str(model.intercept_))
    print("y = ", end="")
    for i in range(len(model.coef_)):
        print(str(round(model.coef_[i], 2)) + " * " + data.columns.values[i], end="")
        if i != len(model.coef_) - 1:
            print(" + ", end="")
    print(" + " + str(round(model.intercept_, 2)))


def save_model(model: LinearRegression, save: bool = True) -> None:
    if save:
        joblib.dump(model, model_path)


def run(save: bool = True) -> None:
    x_train, x_test, y_train, y_test = load_data()
    print("训练集x：" + str(x_train.shape))
    print("训练集y：" + str(y_train.shape))
    model: LinearRegression = build_model()
    fit(model, x_train, y_train)
    print_parameter(model, x_train)
    result = predict(model, x_test)
    print("模型分数：" + str(score(y_test.values, result)))
    show_accuracy(y_test.values, result, save)
    save_model(model, save)


def run(times: int = 1, save: bool = True) -> None:
    max_score = 0
    for i in range(times):
        x_train, x_test, y_train, y_test = load_data()
        if i == 0:
            print("训练集x：" + str(x_train.shape))
            print("训练集y：" + str(y_train.shape))
        print("-------------第" + str(i) + "次-------------")
        model: LinearRegression = build_model()
        fit(model, x_train, y_train)
        print_parameter(model, x_train)
        result = predict(model, x_test)
        model_score = score(y_test.values, result)
        print("模型分数：" + str(score(y_test.values, result)))
        if model_score > max_score:
            max_score = model_score
            show_accuracy(y_test.values, result, save)
            save_model(model, save)
    print("模型分数：" + str(max_score))

if __name__ == "__main__":
    plt.rcParams['figure.dpi'] = plot_dpi
    patch_sklearn()
    run(1000, True)
