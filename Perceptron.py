import matplotlib.pyplot as plt
import numpy as np

class Perceptron_v1(object):
    def __init__(self, num_inputs, learning_rate=0.01, num_epochs=100):
        # Initialisation des poids aléatoirement
        self.weights = np.random.randn(num_inputs + 1)  # +1 pour inclure le biais
        self.learning_rate = learning_rate  # Taux d'apprentissage
        self.num_epochs = num_epochs  # Nombre d'époques d'entraînement

    def activation_function(self, x):
        # Fonction de seuil : retourne 1 si x >= 0, sinon 0
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Calcul de la somme pondérée des entrées
        sum_value = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # Application de la fonction d'activation à la somme
        return self.activation_function(sum_value)

    def train(self, X, y):
        for _ in range(self.num_epochs):  # Itération pour le nombre d'époques spécifié
            for xi, yi in zip(X, y):  # Pour chaque exemple d'entraînement
                prediction = self.predict(xi)  # Faire une prédiction
                error = yi - prediction  # Calculer l'erreur
                # Mise à jour des poids en fonction de l'erreur
                self.weights[1:] += self.learning_rate * error * xi
                self.weights[0] += self.learning_rate * error

def generate_data(num_points, noise=0.1):
    # Génération de données d'entrée aléatoires
    X = np.random.randn(num_points, 2)
    # Création d'étiquettes basées sur une frontière de décision linéaire simple
    y = np.array([1 if x[0] + x[1] > 0 else 0 for x in X])
    # Ajout de bruit aux étiquettes
    y = y.astype(float)
    y += np.random.randn(num_points) * noise
    # Seuillage des étiquettes bruitées à 0 et 1
    y = np.array([1 if yi > 0.5 else 0 for yi in y])
    return X, y

def visualize_data(X, y, perceptron):
    # Tracer les points de données
    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], label='Classe 0')
    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], label='Classe 1')
    
    # Créer une grille pour tracer la frontière de décision
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    
    # Prédire pour chaque point de la grille
    Z = np.array([perceptron.predict(np.array([x, y])) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    # Tracer la frontière de décision
    plt.contourf(xx, yy, Z, alpha=0.4)
    
    # Définir les étiquettes et le titre
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.title('Classification par le Perceptron_v1')
    plt.show()

# Génération des données d'entraînement
X, y = generate_data(100)

# Création et entraînement du perceptron
perceptron = Perceptron_v1(num_inputs=2)
perceptron.train(X, y)

# Visualisation des résultats
visualize_data(X, y, perceptron)

# Génération des données de test
X_test, y_test = generate_data(20)

# Faire des prédictions sur les données de test
predictions = [perceptron.predict(x) for x in X_test]

# Calcul de la précision
accuracy = np.mean(predictions == y_test)
print(f"Précision sur l'ensemble de test : {accuracy:.2f}")


# if needed imports constants from the file like this:
from constants import DATA_PATH

from sklearn.linear_model import Perceptron

class Perceptron_v0(object):
    pass
    """
    This one is made based on the medium.com tutorial here with some adjustments:
    https://medium.com/@becaye-balde/perceptron-building-it-from-scratch-in-python-15716806ef64
    hard to manage typehint on this object so not a priority


    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter


    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def fit(self, X, y):
        # initializing the weights to 0
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        print("Weights:", self.w_)

        # training the model n_iter times
        for _ in range(self.n_iter):
            error = 0

            # loop through each input
            for xi, label in zip(X, y):  # Changed y to label to avoid conflict
                # 1. calculate ŷ (the predicted value)
                y_pred = self.predict(xi)

                # 2. calculate Update
                update = self.eta * (label - y_pred)

                # 3. Update the weights
                self.w_[1:] = self.w_[1:] + update * xi
                print("Updated Weights:", self.w_[1:])

                # Update the bias (X0 = 1)
                self.w_[0] = self.w_[0] + update

                # if update != 0, it means that ŷ != y
                error += int(update != 0.0)

            self.errors_.append(error)

        return self



class Sklearn_Inate_Perceptron():
    def __init__(self, train_data, train_labels):
        # Initialize and train the Perceptron model
        self.perceptron = Perceptron(random_state = 42, max_iter = 20, tol = 0.001)
        self.perceptron.fit(train_data, train_labels)

    def predict(self, test_data):
        # Make predictions using the trained Perceptron model
        return self.perceptron.predict(test_data)




if __name__ == "__main__":
    Perceptron_v0()
