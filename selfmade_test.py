import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from random import choice

DEBUG = False  # Set to True to enable debug prints

# Load mnist dataset
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

def dprint(*args, **kwargs):
    """Debug print function that can be toggled on/off."""
    if DEBUG:  # Set to True to enable debug prints
        print(*args, **kwargs)


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.outputs = []

    def forward(self, x):
        out = np.random.rand(x.shape[0], self.size) < x
        self.outputs.append(out)
        return out
    
    def get_average_output(self):
        if not self.outputs:
            return np.zeros((1, self.size))
        return np.mean(self.outputs, axis=0)
    
    def backward(self, grad_output):
        pass

    def reset(self):
        self.outputs = []

    def __repr__(self):
        return f"InputLayer(size={self.size})"


class Neuron:
    def __init__(self, input_size, threshold=0.7, value=1, learning_rate=0.1):
        self.lr = learning_rate
        self.input_size = input_size
        self.threshold = threshold
        self.value = value
        self.weights = np.random.uniform(-1, 1, input_size)  # Initialize weights uniformly in [-1, 1]
        self.bias = np.random.uniform(-1, 1)  # Initialize bias uniformly in [-1, 1]
        self.outputs = []

    def forward(self, x):
        # Apply the thresholding operation
        weighted_sum = np.dot(x, self.weights) + self.bias
        out = np.where(weighted_sum > self.threshold, self.value, 0)
        self.outputs.append(out)
        return out
    
    def get_average_output(self):
        if not self.outputs:
            return 0
        return np.mean(self.outputs, axis=0)

    def get_average_weight(self):
        if self.weights is None or len(self.weights) == 0:
            return 0
        return np.mean(self.weights)

    def get_average_bias(self):
        if self.bias is None:
            return 0
        return self.bias

    def backward(self, grad_output, avg_input):
        relu_mask = (self.get_average_output() > 0).astype(float)
        d_grad_output = grad_output * relu_mask
        # dL/dW = outer product of upstream_grad and input
        grad_W = np.mean(avg_input * d_grad_output[:, None], axis=0)

        self.weights -= self.lr * grad_W
        self.bias -= self.lr * np.mean(d_grad_output)  # Update bias with the mean of the gradient

        # dL/dx = W^T @ upstream_grad (shape: input_size,)
        grad_x = d_grad_output[:, None] * self.weights[None, :]  # shape (batch, input_size)
        return grad_x

    def reset(self):
        self.outputs = []

    def __repr__(self):
        return f"Neuron(input_size={self.input_size}, threshold={self.threshold}, value={self.value})"


class Layer:
    def __init__(self, input_size, output_size, threshold=0.7, value=1):
        self.neurons = [Neuron(input_size, threshold, value) for _ in range(output_size)]
        self.input_size = input_size
        self.output_size = output_size
        self.activations = []

    def forward(self, x):
        out = np.array([neuron.forward(x) for neuron in self.neurons]).T
        self.activations.append(np.mean(out))
        return out
    
    def get_average_output(self):
        return np.array([neuron.get_average_output() for neuron in self.neurons]).T
    
    def get_average_activation(self):
        return np.mean(self.activations) if self.activations else 0
    
    def get_average_weight(self):
        return np.mean([neuron.get_average_weight() for neuron in self.neurons]) if self.neurons else 0
    
    def get_average_bias(self):
        return np.mean([neuron.get_average_bias() for neuron in self.neurons]) if self.neurons else 0

    def backward(self, grad_output, avg_input):
        next_delta = np.zeros((grad_output.shape[0], self.input_size))
        for i, neuron in enumerate(self.neurons):
            grad_input = neuron.backward(grad_output[:, i], avg_input)
            next_delta += grad_input
        return next_delta
    
    def reset(self):
        self.activations = []
        for neuron in self.neurons:
            neuron.reset()
        
    def __repr__(self):
        return f"Layer(input_size={self.input_size}, output_size={self.output_size}, neurons={len(self.neurons)})"

class SimpleNetwork:
    def __init__(self, input_size, output_size, hidden_size: list[int], iterations=300):
        self.iterations = iterations
        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = InputLayer(input_size)
        self.hidden_layers = []
        for i in range(len(hidden_size)):
            i_size = input_size if i == 0 else hidden_size[i-1]
            o_size = hidden_size[i]
            self.hidden_layers.append(Layer(i_size, o_size, threshold=0.7, value=1))
        self.output_layer = Layer(hidden_size[-1], output_size, threshold=0.7, value=1)

    def forward(self, x):
        outputs = []
        for _ in range(self.iterations):
            nx = self.input_layer.forward(x)
            for layer in self.hidden_layers:
                nx = layer.forward(nx)
            out = self.output_layer.forward(nx)
            outputs.append(out)
        return np.mean(outputs, axis=0)
    
    def backward(self, grad_output):
        # Backward pass through the output layer
        avg_output = self.hidden_layers[-1].get_average_output()
        grad_input = self.output_layer.backward(grad_output, avg_output)
        # Backward pass through the hidden layers
        for l in range(len(self.hidden_layers)-1, -1, -1):
            layer = self.hidden_layers[l]
            if l == 0: # first hidden layer
                avg_input = self.input_layer.get_average_output() # use input layer's average output
            else:
                avg_input = self.hidden_layers[l-1].get_average_output() # use previous layer's average output

            grad_input = layer.backward(grad_input, avg_input)
            

    def get_all_average_outputs(self):
        all_outputs = [self.input_layer.get_average_output()]
        for layer in self.hidden_layers:
            all_outputs.append(layer.get_average_output())
        all_outputs.append(self.output_layer.get_average_output())
        return all_outputs
    
    def get_all_average_activations(self):
        all_activations = []
        for layer in self.hidden_layers:
            all_activations.append(layer.get_average_activation())
        all_activations.append(self.output_layer.get_average_activation())
        return all_activations
    
    def get_all_average_weights(self):
        all_weights = []
        for layer in self.hidden_layers:
            all_weights.append(layer.get_average_weight())
        all_weights.append(self.output_layer.get_average_weight())
        return all_weights
    
    def get_all_average_biases(self):
        all_biases = []
        for layer in self.hidden_layers:
            all_biases.append(layer.get_average_bias())
        all_biases.append(self.output_layer.get_average_bias())
        return all_biases

    def reset(self):
        self.input_layer.reset()
        for layer in self.hidden_layers:
            layer.reset()
        self.output_layer.reset()

    def predict(self, x):
        res = self.forward(x)
        self.reset()  # Reset the model after prediction
        return res

    def __repr__(self):
        layers = '\n\t'.join([layer.__repr__() for layer in self.hidden_layers])
        return f"SimpleNetwork(input_layer={self.input_layer}\nhidden_layers=[\n\t{layers}\n]\noutput_layer={self.output_layer}\n)"
    
def training_step(model:SimpleNetwork, x, y):
    # Forward pass
    output = model.forward(x)
    
    # Compute loss (MSE)
    loss = np.mean((output - y) ** 2)
    
    # Backward pass
    model.backward(output - y)

    if DEBUG:
        print("Average activations:",np.around(model.get_all_average_activations(),3), 
            "Average weights:", np.around(model.get_all_average_weights(),3), 
            "Average biases:", np.around(model.get_all_average_biases(),3)
        )

    # Reset layers for the next iteration
    model.reset()
    
    return loss

def validation_acc_step(model, x, y):
    # Forward pass
    output = model.predict(x)
    # Compute accuracy
    predictions = np.argmax(output, axis=1)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == labels)

    return accuracy

def train_model(model, X, Y, epochs=10, batch_size=32):
    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i + batch_size]
            batch_y = Y[i:i + batch_size]
            
            # Ensure batch_y is in the correct shape
            if len(batch_y.shape) == 1:
                batch_y = batch_y.reshape(-1, 1)
            
            # Perform a training step
            loss = training_step(model, batch_x, batch_y)
            total_loss += loss
                
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (len(X)// batch_size)}, acc: {validation_acc_step(model, X, Y)}")


if __name__ == "__main__":
    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist()
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    input_size = 28 * 28  # MNIST input size
    hidden_size = [10]  # Example hidden layer sizes
    output_size = 10
    iterations = 10

    training_size = 10  # Number of training samples to use for testing

    input_layer = InputLayer(28*28)

    sample_idx = np.random.randint(0, len(train_dataset), size=training_size)  # Randomly select two samples
    train_samples = train_dataset.data.numpy()[sample_idx].reshape(-1, 28 * 28) / (1.2*255.0)  # Normalize the images to [0, 1]
    train_labels = train_dataset.targets.numpy()[sample_idx]  # Get the corresponding labels
    train_labels = np.eye(output_size)[train_labels]  # Convert labels to one-hot encoding
    
    model = SimpleNetwork(input_size=input_size, output_size=output_size, hidden_size=hidden_size, iterations=iterations)
    print(model)

    train_model(model, train_samples, train_labels, epochs=100, batch_size=training_size)

    # Test the model with a sample, show the input image and the predicted label with the correct label
    acc = []
    for i in range(len(train_samples)):

        test_sample = train_samples[i]  # Normalize the image
        test_label = train_labels[i]  # Get the corresponding label

        predicted_output = model.predict(np.array([test_sample]))
        predicted_label = np.argmax(predicted_output, axis=1)[0]  # Get the predicted label
        print(f"Index: {i}, Predicted: {predicted_label}, True: {np.argmax(test_label)}, Output: {predicted_output}")
        acc.append(predicted_label == np.argmax(test_label))

    print("Total accuracy on training set:", np.mean(acc))

        
        