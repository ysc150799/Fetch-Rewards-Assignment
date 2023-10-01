import numpy as np
class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 10000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None    #Weights
        self.b = None    #Bias

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        
        #Default Initialization
        self.W = np.zeros(n_features)  
        self.b = 0
        
        #Gradient Descent
        for i in range(self.epochs):
            Y_predicted = np.dot(X, self.W) + self.b     #y' = w*x + b
            dw = (2 / n_samples) * np.dot(X.T, (Y_predicted - Y))
            db = (2 / n_samples) * np.sum(Y_predicted - Y)

            # update parameters
            self.W -= self.lr * dw
            self.b -= self.lr * db
            
    def predict(self, X):
        Y_predicted = np.dot(X, self.W) + self.b          # type: ignore suppress warning
        return Y_predicted