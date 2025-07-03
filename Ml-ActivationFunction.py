#!pip install numpy==2.0.2
#!pip install matplotlib==3.9.2
import numpy as np
import matplotlib.pyplot as plt
# Sigmoid function and its derivatives
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))
def relu(z):
    return np.maximum(0, z)
### type your answer here
def relu_derivative(z):
    return np.where(z > 0, 1, 0)
# Generate a range of input values
z = np.linspace(-10, 10, 400)
### type your answer here
sigmoid_grad = sigmoid_derivative(z)
relu_grad = relu_derivative(z)
# Plot the activation functions
plt.figure(figsize=(12, 6))

# Plot Sigmoid and its derivative
plt.subplot(1, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid Activation', color='b')
plt.plot(z, sigmoid_grad, label="Sigmoid Derivative", color='r', linestyle='--')
plt.title('Sigmoid Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

# Plot ReLU and its derivative
plt.subplot(1, 2, 2)
plt.plot(z, relu(z), label='ReLU Activation', color='g')
plt.plot(z, relu_grad, label="ReLU Derivative", color='r', linestyle='--')
plt.title('ReLU Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2
z = np.linspace(-5, 5, 100)
tanh_grad=tanh_derivative(z)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(z, tanh(z), label='tanz Activation', color='g')
plt.plot(z, tanh_grad, label="tanz Derivative", color='r', linestyle='--')
plt.title('Tanh Activation & Gradient')
plt.xlabel('Input Value (z)')
plt.ylabel('Activation / Gradient')
plt.legend()

plt.tight_layout()
plt.show()