from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree


def split_data(data: pd.DataFrame) -> list[Any]:
    # TODO: Parametrize which column to classify by
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    return train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )


def rate_model(X_test, y_test, classifier, name) -> None:
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.5f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions))


def train_decision_tree(data: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = split_data(data)

    dtc = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    dtc.fit(X_train, y_train)

    plot_tree(dtc, proportion=True)
    plt.show()

    rate_model(X_test, y_test, dtc, "Decision Tree")


def train_svc(data: pd.DataFrame) -> None:
    X_train, X_test, y_train, y_test = split_data(data)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svc = SVC(kernel="linear", class_weight="balanced")
    svc.fit(X_train_scaled, y_train)

    rate_model(X_test_scaled, y_test, svc, "SVC")


def main() -> None:
    seeds = pd.read_csv("data/seeds_dataset.csv", delimiter=",", header=None)
    train_decision_tree(seeds)
    train_svc(seeds)

    apple_quality = pd.read_csv("data/apple_quality.csv", delimiter=",", header=0)
    apple_quality.drop("A_id", axis=1, inplace=True)
    train_decision_tree(apple_quality)
    train_svc(apple_quality)


if __name__ == "__main__":
    main()
