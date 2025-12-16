import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


