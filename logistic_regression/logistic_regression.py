import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters= n_iters
        self.weight = None
        self.bias = None
        
        self.cost = []

    def fit(self, x, y):
        # init parametes
        n_samples, n_features = x.shape
        self.weight = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            linear_model = np.dot(x, self.weight) + self.bias
            y_predicted = self._sigmoid(linear_model)



            cost = -1 / len(x) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1-y_predicted))
            self.cost.append(cost)


            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        linear_model = np.dot(x, self.weight) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))