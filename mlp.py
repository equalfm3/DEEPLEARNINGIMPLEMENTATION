import numpy as np
from sklearn.metrics import accuracy_score

class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.layer_outputs = [X]
        for i in range(len(self.weights) - 1):
            self.layer_outputs.append(self.relu(np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]))
        self.layer_outputs.append(self.sigmoid(np.dot(self.layer_outputs[-1], self.weights[-1]) + self.biases[-1]))
        return self.layer_outputs[-1]
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y = y.reshape(-1, 1)  # Ensure y is a column vector
        delta = (self.layer_outputs[-1] - y) * self.sigmoid_derivative(self.layer_outputs[-1])
        
        for i in reversed(range(len(self.weights))):
            dW = np.dot(self.layer_outputs[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            if dW.shape != self.weights[i].shape:
                raise ValueError(f"Shape mismatch: dW shape {dW.shape}, weight shape {self.weights[i].shape}")
            
            self.weights[i] -= learning_rate * dW / m
            self.biases[i] -= learning_rate * db / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.layer_outputs[i])
    
    def train(self, X, y, epochs, learning_rate, batch_size=32):
        n_samples = X.shape[0]
        for _ in range(epochs):
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

class MLPWithHistory(MLP):
    def train(self, X, y, epochs, learning_rate, batch_size=32):
        self.loss_history = []
        self.accuracy_history = []
        n_samples = X.shape[0]
        for _ in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            for i in range(0, n_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
                
                loss = np.mean((output - y_batch.reshape(-1, 1)) ** 2)
                accuracy = accuracy_score(y_batch, (output > 0.5).astype(int))
                
                epoch_loss += loss * len(X_batch)
                epoch_accuracy += accuracy * len(X_batch)
            
            self.loss_history.append(epoch_loss / n_samples)
            self.accuracy_history.append(epoch_accuracy / n_samples)
        
        return self.forward(X)