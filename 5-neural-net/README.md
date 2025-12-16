# Neural Networks

## Authors

- Stefan Karczewski (s27459)
- Łukasz Ogorzałek (s27447)

## Running

Make sure you have uv and python, see "Environment Setup" in the main [README.md](../README.md) for more detailed instructions.

When inside the `5-neural-net` directory:

- Wheat seeds dataset: `uv run wheat_seeds.py`
- CIFAR10 dataset: `uv run cifar10.py`
- Fashion MNIST dataset: `uv run fashion.py`
- Kuzushiji-MNIST dataset: `uv run hiragana.py`

## Observations

### Decision Tree/SVM vs Neural Network

- Decision Tree accuracy: 92.06%
- SVM accuracy: 93.65%
- Neural Network accuracy: 95.24%

The neural network with 200 epochs and 2 hidden layers gives a better accuracy than the decision tree and SVM models, but takes much longer to train and requires more computational resources.

## Example Usage

### Wheat Seeds Dataset

Dataset: https://machinelearningmastery.com/standard-machine-learning-datasets/

![Wheat Seeds - Output](./assets/wheat_seeds_output.png)
![Wheat Seeds - Training Loss](./assets/wheat_seeds_loss.png)
![Wheat Seeds - Confusion Matrix](./assets/wheat_seeds_confusion_matrix.png)

### CIFAR10

Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

![CIFAR10 - Output (start)](./assets/cifar10_output_1.png)
![CIFAR10 - Output (end)](./assets/cifar10_output_2.png)
![CIFAR10 - Training Loss](./assets/cifar10_loss.png)
![CIFAR10 - Confusion Matrix](./assets/cifar10_confusion_matrix.png)

### Fashion MNIST

Dataset: https://github.com/zalandoresearch/fashion-mnist

![Fashion - Output (start)](./assets/fashion_output_1.png)
![Fashion - Output (end)](./assets/fashion_output_2.png)
![Fashion - Training Loss](./assets/fashion_loss.png)
![Fashion - Confusion Matrix](./assets/fashion_confusion_matrix.png)

#### Smaller model

- Convolution channel sizes: (32, 64, 128) -> (16, 32, 64)
- Fully connected layer size: 256 -> 128
- Dropout rate: 0.5 -> 0.3

![Fashion (small) - Output (start)](./assets/fashion_small_output_1.png)
![Fashion (small) - Output (end)](./assets/fashion_small_output_2.png)
![Fashion (small) - Training Loss](./assets/fashion_small_loss.png)
![Fashion (small) - Confusion Matrix](./assets/fashion_small_confusion_matrix.png)

### Kuzushiji-MNIST

Dataset: https://github.com/rois-codh/kmnist

![Hiragana - Output (start)](./assets/hiragana_output_1.png)
![Hiragana - Output (end)](./assets/hiragana_output_2.png)
![Hiragana - Training Loss](./assets/hiragana_loss.png)
![Hiragana - Confusion Matrix](./assets/hiragana_confusion_matrix.png)
