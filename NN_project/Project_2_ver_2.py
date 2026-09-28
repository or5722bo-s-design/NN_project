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
        #weight 1&2 randomly around 0 and shapes defined so they are compatible with the layers
        self.W1 = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.input_nodes, self.hidden_nodes))
        self.W2 = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.output_nodes))

    def sigmoid(self, x): #sigmoid function
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_der(self, output): #derivative of the sigmoid function
        return output * (1.0 - output)

    def train_sample(self, x, y): #function that trains one sample
        inputs = np.array(x, ndmin=2).T #make sure input is a 2d array
        targets = np.array(y, ndmin=2).T #make sure output is a 2d array

        #hidden layer
        hidden_inputs = np.dot(self.W1.T, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)

        #output layer
        final_inputs = np.dot(self.W2.T, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        #backpropagation
        output_errors = targets - final_outputs #mse
        hidden_errors = np.dot(self.W2, output_errors)

        self.W2 += self.lr * np.dot((output_errors * self.sigmoid_der(final_outputs)), hidden_outputs.T).T
        self.W1 += self.lr * np.dot((hidden_errors * self.sigmoid_der(hidden_outputs)), inputs.T).T

    def train_batch(self, X, Y):
        #same as above but with matrices
        batch_size = X.shape[1]

        hidden_inputs = np.dot(self.W1.T, X)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(self.W2.T, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = Y - final_outputs
        hidden_errors = np.dot(self.W2, output_errors)

        self.W2 += (self.lr/batch_size) * np.dot(hidden_outputs, (output_errors * self.sigmoid_der(final_outputs)).T)
        self.W1 += (self.lr/batch_size) * np.dot(X, (hidden_errors * self.sigmoid_der(hidden_outputs)).T)

    def sgd(self, x, y, epochs, batch_size, val_x = None, val_y = None):
        num_samples = len(x) #get how many samples we have from input

        for epoch in range(epochs): #we take different perm of data so we don't train with order bias
            permutation = np.random.permutation(num_samples)
            shuffled_x = train_x[permutation]
            shuffled_y = train_y[permutation]

        for i in range(0, num_samples, batch_size): #go through the mini batches
            x_batch = shuffled_x[i : i + batch_size] #slice for the current batch
            y_batch = shuffled_y[i : i + batch_size]

            self.train_batch(x_batch.T, y_batch.T) #train

        if val_x is not None and val_y is not None:
                accuracy = self.evaluate(val_x, val_y)
                print(f"Epoch {epoch + 1}/{epochs} - Validation Accuracy: {accuracy * 100:.2f}%")

    def predict(self, x):
        inputs = np.array(x, ndmin=2).T
        hidden_outputs = self.sigmoid(np.dot(self.W1.T, inputs))
        final_outputs = self.sigmoid(np.dot(self.W2.T, hidden_outputs))
        return final_outputs.T

    def evaluate(self, X, Y_labels):
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        return np.mean(predicted_classes == Y_labels)




nn = NeuralNetwork(input_nodes=784, hidden_nodes=100, output_nodes=10, learning_rate=0.4)

train_y_one = np.eye(10)[train_y]

# for epoch in range(5):
#     for i in range(len(train_x)):
#         nn.train_sample(train_x[i], train_y_one[i])
#     print(f"Epoch {epoch + 1} completed!")

# sample_test_image = test_x[0]
# prediction_scores = nn.predict(sample_test_image)
# predicted_digit = np.argmax(prediction_scores)

# print(f"Predicted Digit: {predicted_digit}, True Label: {test_y[0]}")

nn.sgd(train_x, train_y_one, epochs=10, batch_size=32, val_x=valid_x, val_y=valid_y)