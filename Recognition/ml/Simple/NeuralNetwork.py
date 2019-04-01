import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

class NeuralNetwork:
    def __init__(self, layers, activation='tanh'):
        """
        :param layers: A list containing the number of units in each layers.
        should be at least two values
        :param activation: The activation function to be used. Can be 
        'logistic' or 'tanh'
        """
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i - 1] + 1, layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1, layers[i + 1]))-1)*0.25)

    def fit(self, x, y, learning_rate=0.2, epochs=10000):
        x = np.atleast_2d(x)
        temp = np.ones([x.shape[0], x.shape[1] + 1])
        temp[:, 0:-1] = x # adding the bias unit to the input layer
        x = temp
        y = np.array(y)

        for k in range(epochs):
            i = np.random.randint(x.shape[0])
            a = [x[i]]

            for l in range(len(self.weights)): # going forward network, for each layer
                a.append(self.activation(np.dot(a[l], self.weights[l]))) # Comput the node value
            error = y[i] - a[-1] # Computer the error at the top layer
            deltas = [error * self.activation_deriv(a[-1])] # for output layer, Err calculation (del)

            # Staring backprobagation
            for l in range(len(a) - 2, 0, -1): # we need to begin at the second to last layer
                # Compute the updated error (i,e, deltas) for each node going from top layer to input
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a
                
        
