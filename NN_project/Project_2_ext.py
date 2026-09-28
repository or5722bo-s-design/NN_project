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

ACTIVATIONS = {
    "sigmoid": {
        "fn": lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500))),
        "der": lambda a: a * (1.0 - a)
    },
    "tanh": {
        "fn": lambda x: np.tanh(x),
        "der": lambda a: 1.0 - a**2
    },
    "relu": {
        "fn": lambda x: np.maximum(0, x),
        "der": lambda a: (a > 0).astype(float)
    },
    "leaky_relu": {
        "fn": lambda x: np.where(x > 0, x, 0.01 * x),
        "der": lambda a: np.where(a > 0, 1.0, 0.01)
    }
}

LOSS_FUNCTIONS = {
    "mse": {
        "fn": lambda y_pred, y_true: 0.5 * np.mean(np.sum((y_pred - y_true) ** 2, axis=1)),
        "delta": lambda y_pred, y_true, act_der: (y_pred - y_true) * act_der(y_pred)
    },
    "cross_entropy": {
        "fn": lambda y_pred, y_true: -np.mean(np.sum(
            y_true * np.log(np.clip(y_pred, 1e-15, 1 - 1e-15)) + 
            (1 - y_true) * np.log(np.clip(1 - y_pred, 1e-15, 1 - 1e-15)), 
            axis=1
        )),
        "delta": lambda y_pred, y_true, act_der: (y_pred - y_true)
    }
}

class DeepNeuralNetwork:
    def __init__(self, layer_sizes = [784, 30, 10], learning_rate = 0.1, 
                 output_mode = "one_hot", activation = "sigmoid", custom_act = None, 
                 loss = "mse", custom_loss = None):
        
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.lr = learning_rate
        self.output_mode = output_mode
        self.weights = []

        if custom_act is not None:
            self.act_fn = custom_act["fn"]
            self.act_der = custom_act["der"]
        elif isinstance(activation, str) and activation in ACTIVATIONS:
            self.act_fn = ACTIVATIONS[activation]["fn"]
            self.act_der = ACTIVATIONS[activation]["der"]
        else:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(ACTIVATIONS.keys())} or pass custom_act.")

        if custom_loss is not None:
            self.loss_fn = custom_loss["fn"]
            self.loss_delta_fn = custom_loss["delta"]
            self.loss_name = "custom"
        elif isinstance(loss, str) and loss in LOSS_FUNCTIONS:
            self.loss_fn = LOSS_FUNCTIONS[loss]["fn"]
            self.loss_delta_fn = LOSS_FUNCTIONS[loss]["delta"]
            self.loss_name = loss
        else:
            raise ValueError(f"Unsupported loss: {loss}. Choose from {list(LOSS_FUNCTIONS.keys())}")

        if self.output_mode == "bitwise":
            assert layer_sizes[-1] == 4, "Bitwise mode requires 4 output neurons!"
        elif self.output_mode == "one_hot":
            assert layer_sizes[-1] == 10, "One-hot mode requires 10 output neurons!"
        
        #weights initialized around 0 with scaling factor
        for i  in range(self.num_layers-1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1]
            w = rng.normal(0.0, pow(n_in + 1, -0.5), (n_in + 1, n_out))
            self.weights.append(w)

    def target_mode(self, y_labels):
        if self.output_mode == "one_hot":
            return np.eye(10)[y_labels]
        elif self.output_mode == "bitwise":
            return ((y_labels[:, None] >> np.arange(3, -1, -1)) & 1).astype(np.float32)

    def decode_predictions(self, outputs):
        if self.output_mode == "one_hot":
            return np.argmax(outputs, axis=1)
        elif self.output_mode == "bitwise":
            bi_bits = (outputs >= 0.5).astype(int)
            powers = np.array([8, 4, 2, 1])
            pred_digits = np.dot(bi_bits, powers)
            return pred_digits

    def forward(self, x):
        batch_size = x.shape[1]
        activations = []

        current_activation = np.vstack([x, np.ones((1, batch_size))])
        activations.append(current_activation)

        for i, W in enumerate(self.weights):
            inputs = np.dot(W.T, current_activation)
            outputs = self.act_fn(inputs)

            if i < len(self.weights) - 1:
                current_activation = np.vstack([outputs, np.ones((1, batch_size))])
            else:
                current_activation = outputs
                
            activations.append(current_activation)

        return activations
    
    def train_batch(self, X, Y):
        batch_size = X.shape[1]
        activations = self.forward(X)

        output_activation = activations[-1]
        delta = -self.loss_delta_fn(output_activation, Y, self.act_der)

        deltas = [delta]

        for l in range(len(self.weights) - 1, 0, -1):
            W = self.weights[l]
            error_with_bias = np.dot(W, deltas[0])
            error = error_with_bias[:-1, :]
            hidden_activation = activations[l][:-1, :]
            hidden_delta = error * self.act_der(hidden_activation)
            deltas.insert(0, hidden_delta)

        for m in range(len(self.weights)):
            A_prev = activations[m]
            D_next = deltas[m]
            self.weights[m] += (self.lr / batch_size) * np.dot(A_prev, D_next.T)

    def sgd(self, x, y, epochs, batch_size, val_x=None, val_y=None):
        num_samples = len(x)
        y_encoded = self.target_mode(y)
        val_y_encoded = self.target_mode(val_y) if val_y is not None else None

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            permutation = rng.permutation(num_samples)
            shuffled_x = x[permutation]
            shuffled_y_encoded = y_encoded[permutation]

            for i in range(0, num_samples, batch_size):
                x_batch = shuffled_x[i : i + batch_size].T
                y_batch = shuffled_y_encoded[i : i + batch_size].T

                self.train_batch(x_batch, y_batch)

            if val_x is not None and val_y is not None:
                accuracy = self.evaluate(val_x, val_y)
                val_loss = self.loss(val_x, val_y_encoded)
                print(f"Epoch {epoch + 1}/{epochs} - Val Loss: {val_loss:.4f} - Val Accuracy: {accuracy * 100:.2f}%")

    def predict(self, x):
        inputs = x.T
        activations = self.forward(inputs)
        return activations[-1].T

    def evaluate(self, X, Y_labels):
        predictions = self.predict(X)
        predicted_classes = self.decode_predictions(predictions)
        return np.mean(predicted_classes == Y_labels)

    def loss(self, x, y):
        predictions = self.predict(x)
        loss = 0.5*np.mean(np.sum((predictions-y)**2, axis = 1))
        return loss

def attack(model, x, y, eps = 0.15):
    x_adv = x.copy()
    activations = model.forward(x.T)
    output_activation = activations[-1]

    delta = (output_activation - y.T) * model.act_der(output_activation)

    W1 = model.weights[0]

    W_out = model.weights[-1][:-1, :]
    hidden_activation = activations[1][:-1, :]  
    hidden_delta = np.dot(W_out, delta) * model.act_der(hidden_activation)
    W1 = model.weights[0][:-1, :]
    input_gradient = np.dot(W1, hidden_delta).T
    x_adv = x_adv + eps * np.sign(input_gradient)
    return np.clip(x_adv, 0.0, 1.0)



# dnn_10 = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.5, output_mode="one_hot")
# dnn_10.sgd(train_x, train_y, epochs=10, batch_size=32, val_x=valid_x, val_y=valid_y)

# dnn_4 = DeepNeuralNetwork(layer_sizes=[784, 30, 4], learning_rate=0.5, output_mode= "bitwise")
# dnn_4.sgd(train_x, train_y, epochs=10, batch_size=32, val_x=valid_x, val_y=valid_y)

# # Sigmoid
# nn_sigmoid = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.5, activation="sigmoid")
# nn_sigmoid.sgd(train_x, train_y, epochs=5, batch_size=32, val_x=valid_x, val_y=valid_y)

# # Tanh
# nn_tanh = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.1, activation="tanh")
# nn_tanh.sgd(train_x, train_y, epochs=5, batch_size=32, val_x=valid_x, val_y=valid_y)

# # ReLU
# nn_relu = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.01, activation="relu")
# nn_relu.sgd(train_x, train_y, epochs=5, batch_size=32, val_x=valid_x, val_y=valid_y)

# # mse
# nn_mse = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.5, loss="mse")
# nn_mse.sgd(train_x, train_y, epochs=5, batch_size=32, val_x=valid_x, val_y=valid_y)

# # Cross-Entropy Loss
# nn_ce = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.1, loss="cross_entropy")
# nn_ce.sgd(train_x, train_y, epochs=5, batch_size=32, val_x=valid_x, val_y=valid_y)

epsilon = 0.15

print("Training Initial Base Model")
base_model = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.5, loss="mse")
base_model.sgd(train_x, train_y, epochs=5, batch_size=32, val_x=valid_x, val_y=valid_y)

print("Generating Adversarial Examples from Base Model")
#convert labels for target matching
train_y_encoded = base_model.target_mode(train_y)
valid_y_encoded = base_model.target_mode(valid_y)

#generate attack images for training and validation set
train_x_adv = attack(base_model, train_x, train_y_encoded, eps=epsilon)
valid_x_adv = attack(base_model, valid_x, valid_y_encoded, eps=epsilon)

#check base model accuracy on normal and noisy validation data
acc_clean = base_model.evaluate(valid_x, valid_y)
acc_adv_base = base_model.evaluate(valid_x_adv, valid_y)

print(f"Base Model Accuracy on Clean Data: {acc_clean * 100:.2f}%")
print(f"Base Model Accuracy on Noisy Data: {acc_adv_base * 100:.2f}%")

print("Retraining Model on Normal and Noisy Data")
#combine original images with noisy images
combined_train_x = np.vstack([train_x, train_x_adv])
combined_train_y = np.concatenate([train_y, train_y])

#create model and train on combined data
robust_model = DeepNeuralNetwork(layer_sizes=[784, 30, 10], learning_rate=0.5, loss="mse")
robust_model.sgd(combined_train_x, combined_train_y, epochs=5, batch_size=32, val_x=valid_x, val_y=valid_y)

print("Testing Robustness of Retrained Model")

#test model on old adversarial images
acc_robust_old_adv = robust_model.evaluate(valid_x_adv, valid_y)

#generate new noisy images targeting the robust model directly
valid_x_new_adv = attack(robust_model, valid_x, valid_y_encoded, eps=epsilon)
acc_robust_new_adv = robust_model.evaluate(valid_x_new_adv, valid_y)

print(f"Base Model on Clean Data:             {acc_clean * 100:.2f}%")
print(f"Base Model on Attack:                 {acc_adv_base * 100:.2f}%")
print(f"Robust Model on Base Attack Data:     {acc_robust_old_adv * 100:.2f}%")
print(f"Robust Model on New Targeted Attack:  {acc_robust_new_adv * 100:.2f}%")