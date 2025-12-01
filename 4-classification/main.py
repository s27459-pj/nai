"""
See README.md for running instructions, examples and authors.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def split_data(data: pd.DataFrame) -> list[Any]:
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )


def rate_model(X_test: pd.DataFrame, y_test: pd.Series, classifier, name: str) -> None:
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.5f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))


def train_decision_tree(data: pd.DataFrame, name: str) -> None:
    X_train, X_test, y_train, y_test = split_data(data)

    dtc = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    dtc.fit(X_train, y_train)
    dtc.predict

    rate_model(X_test, y_test, dtc, name)
    plot_tree(dtc, proportion=True)
    plt.show()


def visualize_svm(X_train: pd.DataFrame, y_train: pd.Series, name: str) -> None:
    """
    Visualize decision boundaries of a multi-class SVM

    This is done by splitting the features into two halves and averaging each half.
    """

    n_features = X_train.shape[1]
    mid_point = n_features // 2

    # 2D projection
    X_train_2d = np.column_stack(
        [
            # x: average of second half
            X_train[:, mid_point:].mean(axis=1),
            # y: average of first half
            X_train[:, :mid_point].mean(axis=1),
        ]
    )

    # Encode labels to numeric for visualization
    y_train_colored = LabelEncoder().fit_transform(y_train)

    svc_2d = SVC(kernel="linear", class_weight="balanced")
    svc_2d.fit(X_train_2d, y_train)

    DecisionBoundaryDisplay.from_estimator(
        svc_2d,
        X_train_2d,
        response_method="predict",
        alpha=0.5,
        xlabel=f"Average of features {mid_point + 1}-{n_features}",
        ylabel=f"Average of features 1-{mid_point}",
    )
    plt.scatter(
        x=X_train_2d[:, 0],
        y=X_train_2d[:, 1],
        s=20,
        c=y_train_colored,
        edgecolor="black",
    )
    plt.title(name)
    plt.show()


def train_svm(data: pd.DataFrame, name: str) -> None:
    X_train, X_test, y_train, y_test = split_data(data)

    # SVCs are sensitive to scaling, so we scale the data before training/testing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled: pd.DataFrame = scaler.transform(X_test)  # type: ignore[assignment]

    svc = SVC(kernel="linear", class_weight="balanced")
    svc.fit(X_train_scaled, y_train)

    rate_model(X_test_scaled, y_test, svc, name)
    visualize_svm(X_train_scaled, y_train, name)


def main() -> None:
    # Wheat Seeds dataset
    # https://machinelearningmastery.com/standard-machine-learning-datasets/
    seeds = pd.read_csv("data/wheat_seeds.csv", delimiter=",", header=None)
    train_decision_tree(seeds, "Wheat Seeds - Decision Tree")
    train_svm(seeds, "Wheat Seeds - SVM")

    # Apple Quality dataset
    # https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality
    apple_quality = pd.read_csv("data/apple_quality.csv", delimiter=",", header=0)
    apple_quality.drop("A_id", axis=1, inplace=True)
    train_decision_tree(apple_quality, "Apple Quality - Decision Tree")
    train_svm(apple_quality, "Apple Quality - SVM")


if __name__ == "__main__":
    main()
