import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from Perceptron import Perceptron_v0

def test_Perceptron_v0() -> bool:
    # Load the Iris dataset
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Shuffling the data
    df = shuffle(df)

    # Splitting the data into features (X) and labels (y)
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Only the last column

    # Split the data: 75% for train and 25% for test
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Encoding the labels: 1 for 'Iris-setosa', -1 for all others
    train_labels = np.where(train_labels == 'Iris-setosa', 1, -1)
    test_labels = np.where(test_labels == 'Iris-setosa', 1, -1)

    print('Train data (first 2 samples):', train_data[:2])
    print('Train labels (first 5 labels):', train_labels[:5])
    print('Test data (first 2 samples):', test_data[:2])
    print('Test labels (first 5 labels):', test_labels[:5])

    # Initialize and train the Perceptron
    perceptron = Perceptron_v0(eta=0.1, n_iter=10)
    perceptron.fit(train_data, train_labels)

    # Make predictions on the test data
    y_preds = np.array([perceptron.predict(xi) for xi in test_data])
    print('Predictions:', y_preds)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_preds, test_labels)
    print('Accuracy:', round(accuracy, 2) * 100, "%")

    return accuracy


def test_w_real_data_Perceptron_v0():
    return 1+1


if __name__ == "__main__":
    print(test_Perceptron_v0())
