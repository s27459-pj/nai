# Classification

## Authors

- Stefan Karczewski (s27459)
- Łukasz Ogorzałek (s27447)

## Running

Make sure you have uv and python, see "Environment Setup" in the main [README.md](../README.md) for more detailed instructions.

When inside the `4-classification` directory:

- Run classification: `uv run main.py`

## Observations

- For our datasets, using the `libear` SVM kernel proved to be the most accurate from all available kernels
- Visualizing an SVM is not trivial for more than 2 features, because we want to display it on a 2D scatter plot, which is impossible with more than 2 dimensions out of the box
    - We circumvent this by simplifying the dataset into 2D, training another SVM on the simplified dataset and graphing it

## Example Usage

### Wheat Seeds Dataset

Dataset: https://machinelearningmastery.com/standard-machine-learning-datasets/

![Wheat Seeds - Rating](./assets/wheat_seeds_rating.png)
![Wheat Seeds - Decision Tree Visualization](./assets/wheat_seeds_decision_tree.png)
![Wheat Seeds - SVM Visualization](./assets/wheat_seeds_svm.png)

### Apple Quality Dataset

Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality

![Apple Quality - Rating](./assets/apple_quality_rating.png)
![Apple Quality - Decision Tree Visualization](./assets/apple_quality_decision_tree.png)
![Apple Quality - SVM Visualization](./assets/apple_quality_svm.png)
