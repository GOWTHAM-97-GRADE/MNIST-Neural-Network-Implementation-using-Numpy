
# MNIST Neural Network Implementation

A Python and NumPy-based implementation of a feedforward neural network for the MNIST dataset, featuring data conversion, network training, and performance evaluation.

This repository contains an implementation of a neural network for the MNIST dataset using Python and NumPy. The project includes data conversion from binary to CSV format, a neural network class definition, and training and evaluation scripts.

## Overview

The code defines a simple feedforward neural network with two hidden layers. Key functionalities include:

- **Data Conversion**: Converts MNIST data from binary format to CSV format.
- **Neural Network Class**: Defines a simple neural network with initialization, forward propagation, backward propagation, weight updates, and accuracy calculation.
- **Training Script**: Trains the neural network using the converted MNIST data and evaluates its performance.

## Acknowledgment

This implementation is inspired by educational examples of neural network implementations and the principles of gradient-based optimization.

## Components

### Data Conversion

The `convert` function converts MNIST data from binary format to CSV format. This is essential for preparing the data for training and testing the neural network.

### Neural Network Class

The `NeuralNetwork` class defines a simple feedforward neural network with two hidden layers. Key components include:

- **Initialization**: Initializes the network parameters (weights) with random values.
- **Activation Functions**: Implements sigmoid and softmax activation functions.
- **Forward Propagation**: Computes the network output.
- **Backward Propagation**: Computes gradients and updates weights.
- **Accuracy Calculation**: Evaluates the network's accuracy on test data.

### Training Script

The training script reads the converted MNIST data, initializes the neural network, and trains it using the specified number of epochs and learning rate. It prints the accuracy after each epoch.

## References

The design and implementation of this code are inspired by various educational resources on neural networks and gradient-based optimization.

## Usage

### Define Hyperparameters

Set the desired hyperparameters for training the neural network:

```python
learning_rate = 0.1
epochs = 15
```

### Convert Data

Convert the MNIST data from binary format to CSV format using the provided `convert` function:

```python
mnist_train_x = "path/to/train-images-idx3-ubyte"
mnist_train_y = "path/to/train-labels-idx1-ubyte"
mnist_test_x = "path/to/t10k-images-idx3-ubyte"
mnist_test_y = "path/to/t10k-labels-idx1-ubyte"

convert(mnist_train_x, mnist_train_y, "path/to/train.csv", 60000)
convert(mnist_test_x, mnist_test_y, "path/to/test.csv", 10000)
```

### Initialize and Train the Network

Initialize the neural network and train it using the converted data:

```python
train_file = open("path/to/train.csv", "r")
train_list = train_file.readlines()
train_file.close()

test_file = open("path/to/test.csv", "r")
test_list = test_file.readlines()
test_file.close()

dnn = NeuralNetwork(model=[784, 128, 64, 10], epochs=epochs, learning_rate=learning_rate)
dnn.train(train_list, test_list)
```

### Make Predictions and Evaluate Accuracy

After training, make predictions on test data and evaluate the network's accuracy:

```python
accuracy = dnn.accuracy(test_list)
print(f"Final Accuracy: {accuracy * 100}%")
```

## Running the Code

To run the example, create a new Python script (e.g., `train_mnist.py`) and include the provided code. Execute the script in your terminal:

```bash
python train_mnist.py
```

## Acknowledgments

The design and implementation of this code are inspired by various educational resources on neural networks and gradient-based optimization. Special thanks to the creators of these resources for providing foundational knowledge and inspiration.

## References

- MNIST Dataset: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

---
