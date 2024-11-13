import pytest
import numpy as np
import pandas as pd
from Perceptron import Perceptron_v1

# Charger les données Iris
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, names=column_names)

# Préparation des données
X = iris_data.iloc[:, :4].values
y = (iris_data['class'] == 'Iris-setosa').astype(int).values

@pytest.fixture
def perceptron():
    return Perceptron_v1(num_inputs=4)

def test_perceptron_initialization(perceptron):
    assert len(perceptron.weights) == 5  # 4 entrées + 1 biais
    assert perceptron.learning_rate == 0.01
    assert perceptron.num_epochs == 100

def test_activation_function(perceptron):
    assert perceptron.activation_function(1) == 1
    assert perceptron.activation_function(-1) == 0
    assert perceptron.activation_function(0) == 1

def test_predict(perceptron):
    perceptron.weights = np.array([0.5, 1, -1, 0.5, -0.5])  # Biais, poids1, poids2, poids3, poids4
    assert perceptron.predict(np.array([5.1, 3.5, 1.4, 0.2])) == 1
    assert perceptron.predict(np.array([6.3, 3.3, 6.0, 2.5])) == 0

def test_train():
    perceptron = Perceptron_v1(num_inputs=4, num_epochs=1000)
    perceptron.train(X, y)
    predictions = [perceptron.predict(xi) for xi in X]
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.9  # On s'attend à une précision d'au moins 90% pour la classification Iris-setosa vs. autres

def test_accuracy():
    perceptron = Perceptron_v1(num_inputs=4)
    perceptron.train(X, y)
    predictions = [perceptron.predict(xi) for xi in X]
    accuracy = np.mean(predictions == y)
    assert accuracy > 0.9  # On s'attend à une précision d'au moins 90%