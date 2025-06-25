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
    def __init__(self, threshold=0.7, value=1, learning_rate=0.01):
        self.lr = learning_rate
        self.threshold = threshold
        self.value = value
        self.outputs = []

    def __setup__(self, input_size, self_size, post_size):
        """Setup method to initialize weights and bias."""
        self.input_size = input_size
        self.self_size = self_size
        self.post_size = post_size

        self.weights = np.random.uniform(-1, 1, self.input_size + self.self_size + post_size)
        self.bias = np.random.uniform(-1, 1)

        self.input_size = input_size
        self.self_size = self_size
        self.post_size = post_size

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

    def backward(self, grad_output, avg_input):
        relu_mask = (self.get_average_output() > 0).astype(float)
        d_grad_output = grad_output * relu_mask
        # dL/dW = outer product of upstream_grad and input
        grad_W = np.mean(avg_input * d_grad_output[:, None], axis=0)

        self.weights -= self.lr * grad_W
        self.bias -= self.lr * np.mean(d_grad_output)  # Update bias with the mean of the gradient

        # dL/dx = W^T @ upstream_grad (shape: input_size,)
        grad_x = d_grad_output[:, None] * self.weights[None, :]  # shape (batch, input_size)
        # remove self_size part of the weights
        grad_x = grad_x[:, :self.input_size]  # Keep only the input_size
        return grad_x

    def reset(self):
        self.outputs = []

    def __repr__(self):
        return f"Neuron(input_size={self.input_size}, threshold={self.threshold}, value={self.value})"


class Layer:
    def __init__(self, size, threshold=0.7, value=1):
        self.last_activation = None  # Initialize last activation to zeros
        self.output_size = size
        self.threshold = threshold
        self.value = value

    def __setup__(self, input_size, post_size):
        """Setup method to initialize neurons."""
        self.input_size = input_size
        self.post_size = post_size

        self.neurons = []
        for _ in range(self.output_size):
            neuron = Neuron(self.threshold, self.value)
            neuron.__setup__(input_size, self.output_size, post_size)
            self.neurons.append(neuron)
            

    def forward(self, x, post_layer=None):
        if post_layer is None:
            post_layer = np.zeros((x.shape[0], self.post_size))

        if self.last_activation is None:
            self.last_activation = np.zeros((x.shape[0], self.output_size))
    
        combined_input = np.hstack((x, self.last_activation,post_layer))  # Combine input with last activation
        out = np.array([neuron.forward(combined_input) for neuron in self.neurons]).T
        self.last_activation = out
        return out
    
    def get_average_output(self):
        return np.array([neuron.get_average_output() for neuron in self.neurons]).T
    
    def get_latest_activation(self):
        return self.last_activation
    
    def backward(self, grad_output, pre_avg_input):
        self_avg_output = self.get_average_output()
        avg_input = np.hstack((pre_avg_input, self_avg_output))  # Combine input with average output of the layer

        next_delta = np.zeros((grad_output.shape[0], self.input_size))
        for i, neuron in enumerate(self.neurons):
            grad_input = neuron.backward(grad_output[:, i], avg_input)
            next_delta += grad_input
        return next_delta
    
    def reset(self):
        self.activations = []
        self.last_activation = None  # Reset last activation
        for neuron in self.neurons:
            neuron.reset()
        
    def __repr__(self):
        return f"Layer(input_size={self.input_size}, output_size={self.output_size}, neurons={len(self.neurons)})"

class SimpleNetwork:
    def __init__(self, input_size, output_size, hidden_layers: list[Layer], iterations=300):
        self.iterations = iterations
        self.input_size = input_size
        self.output_size = output_size

        self.input_layer = InputLayer(input_size)

        self.hidden_layers = []
        match len(hidden_layers):
            case 0:
                raise ValueError("At least one hidden layer is required.")
            case 1: # because input is input layer and output is output layer
                layer = hidden_layers[0]
                layer.__setup__(input_size, output_size)
                self.hidden_layers.append(layer)

            case 2: # first has input layer, second has output layer
                layer = hidden_layers[0]
                layer.__setup__(input_size, hidden_layers[1].output_size)
                self.hidden_layers.append(layer)
                layer2 = hidden_layers[1]
                layer2.__setup__(hidden_layers[0].output_size, output_size)
                self.hidden_layers.append(layer2)
            case _: # more than two hidden layers, normal case
                first_layer = hidden_layers[0]
                first_layer.__setup__(input_size, hidden_layers[1].output_size, hidden_layers[1].output_size)
                self.hidden_layers.append(first_layer)
                # Setup the rest of the hidden layers
                for i in range(len(1, hidden_layers[1:-1])):
                    layer = hidden_layers[i]
                    layer.__setup__(hidden_layers[i-1].output_size, hidden_layers[i+1].output_size)
                    self.hidden_layers.append(layer)
                # Setup the last hidden layer
                last_layer = hidden_layers[-1]
                last_layer.__setup__(hidden_layers[-2].output_size, output_size)
                self.hidden_layers.append(last_layer)

        self.output_layer = Layer(output_size)
        self.output_layer.__setup__(hidden_layers[-1].output_size, 0)

    def forward(self, x):
        outputs = []
        layer_outputs = []
        for i in range(self.iterations):
            layer_outputs.append([])
            nx = self.input_layer.forward(x)
            for j, layer in enumerate(self.hidden_layers):
                if i == 0: # if first iteration, post layer is zeros
                    if j == len(self.hidden_layers) - 1: # if it is the last layer, use output layer
                        post_layer = np.zeros((nx.shape[0], self.output_size))
                        nx = layer.forward(nx, post_layer)
                    else: # if it is not the last layer, use next layer's size
                        post_layer = np.zeros((nx.shape[0], self.hidden_layers[j+1].output_size))
                        nx = layer.forward(nx, post_layer)
                else: # if not first iteration, use next layer's output from last iteration
                    post_layer = layer_outputs[i-1][j+1]
                    nx = layer.forward(nx, post_layer)
                layer_outputs[i].append(nx)


            out = self.output_layer.forward(nx)
            layer_outputs[i].append(out)
            outputs.append(out)
        return np.mean(outputs, axis=0)
    
    def backward(self, grad_output):
        # Backward pass through the output layer
        avg_output = self.hidden_layers[-1].get_average_output()
        grad_input = self.output_layer.backward(grad_output, avg_output)
        # Backward pass through the hidden layers
        for l in range(len(self.hidden_layers)-1, -1, -1):
            layer = self.hidden_layers[l]
            pre_avg_input = None
            post_avg_input = None
            if l == 0: # first hidden layer
                pre_avg_input = self.input_layer.get_average_output() # use input layer's average output
                if l == len(self.hidden_layers) - 1: # if it is the last layer, use output layer's average output
                    post_avg_input = self.output_layer.get_average_output()
                else: # if it is not the last layer, use next layer's average output
                    post_avg_input = self.hidden_layers[l+1].get_average_output()
            elif l == len(self.hidden_layers) - 1: # last hidden layer
                pre_avg_input = self.hidden_layers[l-1].get_average_output() # use previous
                post_avg_input = self.output_layer.get_average_output()
            
            else:
                pre_avg_input = self.hidden_layers[l-1].get_average_output() # use previous layer's average output
                post_avg_input = self.hidden_layers[l+1].get_average_output() # use next layer's average output
            
            avg_input = np.hstack((pre_avg_input, post_avg_input))  # Combine
            grad_input = layer.backward(grad_input, avg_input)
            

    def get_all_average_outputs(self):
        all_outputs = [self.input_layer.get_average_output()]
        for layer in self.hidden_layers:
            all_outputs.append(layer.get_average_output())
        all_outputs.append(self.output_layer.get_average_output())
        return all_outputs
    
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
    output_size = 10
    iterations = 20

    training_size = 60000  # Number of training samples to use for testing

    sample_idx = np.random.randint(0, len(train_dataset), size=training_size)  # Randomly select two samples
    train_samples = train_dataset.data.numpy()[sample_idx].reshape(-1, 28 * 28) / (1.2*255.0)  # Normalize the images to [0, 1]
    train_labels = train_dataset.targets.numpy()[sample_idx]  # Get the corresponding labels
    train_labels = np.eye(output_size)[train_labels]  # Convert labels to one-hot encoding
    
    model = SimpleNetwork(input_size=input_size, output_size=output_size, hidden_layers=[
        Layer(size=10, threshold=0.7, value=1),
    ], iterations=iterations)
    print(model)

    train_model(model, train_samples, train_labels, epochs=100, batch_size=1024)

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
