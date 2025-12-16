"""
See README.md for running instructions, examples and authors.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from util import plot_training_loss

RANDOM_SEED = 42
TEST_SIZE = 0.3
BATCH_SIZE = 16
LEARNING_RATE = 0.01
NUM_EPOCHS = 200  # 200 is good enough - 100 is too low, 300 gives the same accruracy
HIDDEN_LAYER_1_SIZE = 64
HIDDEN_LAYER_2_SIZE = 32
DROPOUT_RATE = 0.2


class WheatSeedsClassifier(nn.Module):
    """Neural network for classifying wheat seed varieties"""

    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes

        self.linear_1 = nn.Linear(input_size, HIDDEN_LAYER_1_SIZE)
        self.linear_2 = nn.Linear(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE)
        self.linear_3 = nn.Linear(HIDDEN_LAYER_2_SIZE, num_classes)
        self.norm_1 = nn.BatchNorm1d(HIDDEN_LAYER_1_SIZE)
        self.norm_2 = nn.BatchNorm1d(HIDDEN_LAYER_2_SIZE)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        steps: list[nn.Module] = [
            # Input -> hidden layer 1
            self.linear_1,
            # Hidden layer 1,
            self.norm_1,
            self.relu,
            self.dropout,
            # Hidden layer 1 -> hidden layer 2,
            self.linear_2,
            # Hidden layer 2,
            self.norm_2,
            self.relu,
            self.dropout,
            # Hidden layer 2 -> output
            self.linear_3,
        ]

        for module in steps:
            x = module(x)

        return x

    def describe(self) -> None:
        """Print model architecture and parameters"""

        print("Model Architecture:")
        print(f"  Input features: {self.input_size}")
        print(f"  Hidden layer 1: {HIDDEN_LAYER_1_SIZE}")
        print(f"  Hidden layer 2: {HIDDEN_LAYER_2_SIZE}")
        print(f"  Output classes: {self.num_classes}")
        print(f"  Dropout rate: {DROPOUT_RATE}")
        print()

        print("Training Parameters:")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Epochs: {NUM_EPOCHS}")
        print()


def split_data(data: pd.DataFrame) -> list[Any]:
    """
    Split input dataset into training and testing sets

    Use the last column as the target variable and the rest as features.
    """

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )


def prepare_data_loaders(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[DataLoader, DataLoader]:
    """Scale data and create PyTorch DataLoaders"""

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train - 1)  # Convert classes 1,2,3 to 0,1,2
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test - 1)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int = NUM_EPOCHS,
) -> list[float]:
    """Train the neural network and return training loss history"""

    model.train()
    loss_history: list[float] = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the model on test data and print metrics"""

    model.eval()
    all_predictions: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(y_batch.numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Convert back to original class labels (1, 2, 3)
    predictions_original = predictions + 1
    targets_original = targets + 1

    accuracy = accuracy_score(targets, predictions)

    print(f"Accuracy: {accuracy:.5f}")
    print("\nClassification Report:")
    print(
        classification_report(
            targets_original,
            predictions_original,
            target_names=["Class 1", "Class 2", "Class 3"],
        )
    )

    return predictions, targets


def plot_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> None:
    """Plot confusion matrix for the predictions"""

    # Convert back to original class labels
    predictions_original = predictions + 1
    targets_original = targets + 1

    cm = confusion_matrix(targets_original, predictions_original)

    _, ax = plt.subplots(figsize=(10, 8))

    classes = ["Class 1", "Class 2", "Class 3"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        square=True,
        cbar_kws={"label": "Count"},
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
        annot_kws={"size": 14, "weight": "bold"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.show()


def train_neural_network(data: pd.DataFrame) -> None:
    """Train and evaluate a neural network on the wheat seeds dataset"""

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Split and prepare data
    X_train, X_test, y_train, y_test = split_data(data)
    train_loader, test_loader = prepare_data_loaders(X_train, X_test, y_train, y_test)

    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    model = WheatSeedsClassifier(input_size, num_classes)
    model.describe()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    loss_history = train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)

    predictions, targets = evaluate_model(model, test_loader)

    plot_training_loss(loss_history)
    plot_confusion_matrix(predictions, targets)


def main() -> None:
    seeds = pd.read_csv("data/wheat_seeds.csv", delimiter=",", header=0)

    print("Wheat Seeds Dataset:")
    print(f"  Shape: {seeds.shape}")
    print(f"  Features: {list(seeds.columns[:-1])}")
    print(f"  Classes: {sorted(seeds['class'].unique())}")
    print(f"  Samples per class: {seeds['class'].value_counts().to_dict()}")
    print()

    train_neural_network(seeds)


if __name__ == "__main__":
    main()
