import numpy as np
import pickle
import gzip

with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y = test_set

seed = 11
rng = np.random.default_rng(seed)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        
        #weights initialized around 0 with scaling factor
        self.W1 = rng.normal(0.0, pow(self.input_nodes+1, -0.5), (self.input_nodes+1, self.hidden_nodes))
        self.W2 = rng.normal(0.0, pow(self.hidden_nodes+1, -0.5), (self.hidden_nodes+1, self.output_nodes))

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_der(self, output):
        return output * (1.0 - output)

    def train_sample(self, x, y):
        inputs = np.array(x, ndmin=2).T
        targets = np.array(y, ndmin=2).T

        #hidden layer
        hidden_inputs = np.dot(self.W1.T, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(self.W2.T, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        #backpropagation
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.W2, output_errors)

        #weights
        self.W2 += self.lr * np.dot(hidden_outputs, (output_errors * self.sigmoid_der(final_outputs)).T)
        self.W1 += self.lr * np.dot(inputs, (hidden_errors * self.sigmoid_der(hidden_outputs)).T)

    def train_batch(self, X, Y):
        batch_size = X.shape[1]

        X_bias = np.vstack([X, np.ones((1, batch_size))])

        hidden_inputs = np.dot(self.W1.T, X_bias)
        hidden_outputs = self.sigmoid(hidden_inputs)

        hidden_outputs_bias = np.vstack([hidden_outputs, np.ones((1, batch_size))])

        final_inputs = np.dot(self.W2.T, hidden_outputs_bias)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = Y - final_outputs
        output_grad = output_errors * self.sigmoid_der(final_outputs)
        hidden_errors = np.dot(self.W2, output_grad)[:-1,:]
        hidden_grad = hidden_errors * self.sigmoid_der(hidden_outputs)

        self.W2 += (self.lr / batch_size) * np.dot(hidden_outputs_bias, output_grad.T)
        self.W1 += (self.lr / batch_size) * np.dot(X_bias, hidden_grad.T)

    def sgd(self, x, y, epochs, batch_size, val_x=None, val_y=None):
        num_samples = len(x)
        val_y_one = np.eye(10)[val_y] if val_y is not None else None

        for epoch in range(epochs):
            permutation = rng.permutation(num_samples)
            shuffled_x = x[permutation]
            shuffled_y = y[permutation]

            for i in range(0, num_samples, batch_size):
                x_batch = shuffled_x[i : i + batch_size].T
                y_batch = shuffled_y[i : i + batch_size].T

                self.train_batch(x_batch, y_batch)

            if val_x is not None and val_y is not None:
                accuracy = self.evaluate(val_x, val_y)
                val_loss = self.loss(val_x, val_y_one)
                print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f} - Val Accuracy: {accuracy * 100:.2f}%")

    def predict(self, x):
        #accepts single image vector or full data matrix
        inputs = np.array(x, ndmin=2).T
        num_samples = len(x)

        inputs_bias = np.vstack([inputs, np.ones((1, num_samples))])
        hidden_outputs = self.sigmoid(np.dot(self.W1.T, inputs_bias))
        hidden_bias = np.vstack([hidden_outputs, np.ones((1, num_samples))])
        final_outputs = self.sigmoid(np.dot(self.W2.T, hidden_bias))
        return final_outputs.T

    def evaluate(self, X, Y_labels):
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return np.mean(predicted_classes == Y_labels)

    def loss(self, x, y):
        predictions = self.predict(x)
        loss = 0.5*np.mean(np.sum((predictions-y)**2, axis = 1))
        return loss

    def attack(self, x, y, eps = 0.15):
        num_samples = len(x)
        x_batch = x.T
        y_batch = y.T

        x_bias = np.vstack([x_batch, np.ones((1, num_samples))])
        hidden_inputs = np.dot(self.W1.T, x_bias)
        hidden_outputs = self.sigmoid(hidden_inputs)

        hidden_outputs_bias = np.vstack([hidden_outputs, np.ones((1, num_samples))])
        final_inputs = np.dot(self.W2.T, hidden_outputs_bias)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = y_batch - final_outputs
        output_grad = output_errors * self.sigmoid_der(final_outputs)

        hidden_errors_bias = np.dot(self.W2, output_grad)[:-1, :]
        hidden_grad = hidden_errors_bias * self.sigmoid_der(hidden_outputs)

        dx = -np.dot(self.W1[:-1, :], hidden_grad) #attack
        x_adv = x_batch + eps * np.sign(dx) #create adversary image
        x_adv = np.clip(x_adv, 0.0, 1.0) #clip pixels, so the image will be the appropriate size
        return x_adv.T



#execute
nn = NeuralNetwork(input_nodes=784, hidden_nodes=100, output_nodes=10, learning_rate=0.4)

#encode targets
train_y_one = np.eye(10)[train_y]

#train using sgd
nn.sgd(train_x, train_y_one, epochs=10, batch_size=32, val_x=valid_x, val_y=valid_y)

#evaluate clean test accuracy
clean_acc = nn.evaluate(test_x, test_y)
print(f"Clean Test Accuracy: {clean_acc * 100:.2f}%")

#generate noisy images on test set
test_y_one = np.eye(10)[test_y]
test_x_adv = nn.attack(test_x, test_y_one, eps=0.15)

#evaluate noisy test accuracy
adv_acc = nn.evaluate(test_x_adv, test_y)
print(f"Noisy Test Accuracy (epsilon=0.15): {adv_acc * 100:.2f}%")