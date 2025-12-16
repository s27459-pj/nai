"""
See README.md for running instructions, examples and authors.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from util import evaluate_model, plot_confusion_matrix, plot_training_loss, train_model

RANDOM_SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
CONV3_CHANNELS = 128
FC1_SIZE = 256
DROPOUT_RATE = 0.5


class FashionMNISTClassifier(nn.Module):
    """Convolutional neural network for classifying Fashion-MNIST images"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, CONV1_CHANNELS, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, kernel_size=3, padding=1)

        # Batch normalization
        self.norm1 = nn.BatchNorm2d(CONV1_CHANNELS)
        self.norm2 = nn.BatchNorm2d(CONV2_CHANNELS)
        self.norm3 = nn.BatchNorm2d(CONV3_CHANNELS)

        # Pooling and activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

        # Fully connected layers
        # After 3 pooling layers: 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc1 = nn.Linear(CONV3_CHANNELS * 3 * 3, FC1_SIZE)
        self.fc2 = nn.Linear(FC1_SIZE, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1: 28x28x1 -> 14x14x32
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 2: 14x14x32 -> 7x7x64
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Conv block 3: 7x7x64 -> 3x3x128
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten: 3x3x128 -> 1152
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def describe(self) -> None:
        """Print model architecture and parameters"""

        print("Model Architecture:")
        print("  Input: 28x28x1 (Grayscale images)")
        print(f"  Conv layer 1: {CONV1_CHANNELS} filters")
        print(f"  Conv layer 2: {CONV2_CHANNELS} filters")
        print(f"  Conv layer 3: {CONV3_CHANNELS} filters")
        print(f"  FC layer 1: {FC1_SIZE} units")
        print(f"  Output classes: {self.num_classes}")
        print(f"  Dropout rate: {DROPOUT_RATE}")
        print()

        print("Training Parameters:")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Epochs: {NUM_EPOCHS}")
        print()


def prepare_data_loaders() -> tuple[DataLoader, DataLoader]:
    """Download Fashion-MNIST dataset and create PyTorch DataLoaders"""

    # Normalization values for Fashion-MNIST
    mean = (0.2860,)
    std = (0.3530,)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Training data
    train_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, test_loader


def train_neural_network(
    train_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
) -> None:
    """Train and evaluate a neural network on the Fashion-MNIST dataset"""

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # Initialize model
    model = FashionMNISTClassifier(num_classes=len(class_names))
    model = model.to(device)
    model.describe()

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    loss_history = train_model(
        model,
        train_loader,
        criterion,
        optimizer,
        NUM_EPOCHS,
        device,
        print_interval=1,
    )

    predictions, targets, accuracy = evaluate_model(model, test_loader, device)

    print(f"Accuracy: {accuracy:.5f}")
    print("\nClassification Report:")
    print(
        classification_report(
            targets,
            predictions,
            target_names=class_names,
        )
    )

    plot_training_loss(loss_history)
    plot_confusion_matrix(predictions, targets, class_names)


def main() -> None:
    # Fashion-MNIST class labels
    class_names = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    print("Fashion-MNIST Dataset:")
    print(f"  Classes: {class_names}")
    print("  Training samples: 60,000")
    print("  Test samples: 10,000")
    print("  Image size: 28x28x1 (Grayscale)")
    print()

    train_loader, test_loader = prepare_data_loaders()
    train_neural_network(train_loader, test_loader, class_names)


if __name__ == "__main__":
    main()
