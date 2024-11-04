import numpy as np

# if needed imports constants from the file like this:
from constants import DATA_PATH

from sklearn.linear_model import Perceptron

class Perceptron_v0(object):
    pass
    """
    This one is made based on the medium.com tutorial here:
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
