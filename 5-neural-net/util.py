import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def plot_training_loss(loss_history: list[float]) -> None:
    """Plot the training loss over epochs"""

    _, ax = plt.subplots(figsize=(12, 6))

    loss_df = pd.DataFrame(
        {
            "epoch": range(1, len(loss_history) + 1),
            "loss": loss_history,
        }
    )

    sns.lineplot(
        data=loss_df,
        x="epoch",
        y="loss",
        linewidth=2.5,
        color="#2E86AB",
        ax=ax,
    )

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax.set_title("Training Loss Over Time", fontsize=14, fontweight="bold", pad=20)

    ax.fill_between(loss_df["epoch"], loss_df["loss"], alpha=0.3, color="#2E86AB")

    plt.tight_layout()
    plt.show()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device | None = None,
    print_interval: int = 10,
) -> list[float]:
    """Train a neural network and return training loss history"""

    model.train()
    loss_history: list[float] = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            if device is not None:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

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

        if (epoch + 1) % print_interval == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return loss_history
