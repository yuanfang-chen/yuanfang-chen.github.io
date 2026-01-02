---
layout: post
title:  "Deep Learning Notes"
# date:   2025-12-08 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---

** Focus and inferencing for recommendation**

## Overview

A standard feedforward neural network with 3 hidden layers.

![Diagram of a neural network with three hidden layers: input layer, multiple hidden layers, output layer](https://assets.ibm.com/is/image/ibm/deep-neural-network-diagram:2x1?dpr=on%2C2&wid=320&hei=160)

The intermediate layers, called the network’s *hidden layers*, are where most of the learning occurs. It’s the inclusion of *multiple* hidden layers that distinguishes a deep learning model from a “non-deep” neural network, such as a [restricted Boltzmann machine (RBN)](https://developer.ibm.com/tutorials/build-a-recommendation-engine-with-a-restricted-boltzmann-machine-using-tensorflow/) or standard multilayer perceptron (MLP). The presence of multiple hidden layers allows a deep learning model to learn complex hierarchical features of data, with earlier layers identifying broader patterns and deeper layers identifying more granular patterns.

## Multilayer Perceptron (MLP)
This is also known as fully connected or FC.

However, despite practitioners’ effort to train high performing models, neural networks still face challenges similar to other machine learning models—most significantly, overfitting. When a neural network becomes overly complex with too many parameters, the model will overfit to the training data and predict poorly. Overfitting is a common problem in all kinds of neural networks, and paying close attention to [bias-variance tradeoff](https://www.ibm.com/think/topics/bias-variance-tradeoff) is paramount to creating high-performing neural network models.  

## activation function

### [softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

### Loss function (aka cost function, error function etc)

The goal of this loss function is to quantify inaccuracy in a way that appropriately reflects both the nature and magnitude of the error of the model’s output for each input. Different mathematical formulas for loss are best suited to specific tasks: for example, variants of *mean squared error* work well for regression problems, whereas variants of *cross-entropy loss* work well for classification.



## Training deep neural networks

### Backpropagation

**Why Backpropagation**

As Nielsen explains, one can easily estimate the impact of changes to any specific weight *w*j in the network by simply completing a forward pass for two slightly different values of *w*j, while keeping all other parameters unchanged, and comparing the resulting loss for each pass. By formalizing that process into a straightforward equation and implementing a few lines of code in Python, you can automate that process for each weight in the network.

But now imagine that there are 1 million weights in your model, which would be quite modest for a modern deep learning model. To compute the entire gradient, you’d need to complete 1,000,001 forward passes through the network: 1 to establish a baseline, and then another pass to evaluate changes to each of the million weights.

Backpropagation can achieve the same goal in *2* passes: 1 forward pass and 1 backward pass.



We can therefore use the "chain rule", a [calculus principle dating back to the 17th century](https://www.jstor.org/stable/27900650), to compute the rate at which each neuron contributes to overall loss. In doing so, we can calculate the impact of changes to any variable—that is, to any weight or bias—within the equations those neurons represent.

### Key mathematical concepts for backpropagation



To simplify an explanation of how backpropagation works, it will be helpful to first briefly review some core mathematical concepts and terminology.

- A **derivative** is the rate of change in an equation at a specific instant. In a linear equation, the rate of change is a constant slope. In a *nonlinear* equation, like those used for activation functions, this slope varies. **Differentiation** is the process of finding the derivative of a specific function. By differentiating a nonlinear function, we can then find the slope—its instantaneous rate of change—at any specific point in the curve.

- In functions with multiple variables, a **partial derivative** is the derivative of one variable concerning the others. If we change one variable, but keep the others the same, how does the output of the overall function change? The activation functions of individual nodes in a neural network have many variables, including the many inputs from neurons in previous layers and the weights applied to those inputs. When dealing with a specific node *n*, finding the partial derivatives of the activation functions of neurons from the previous layer allows us to isolate the impact of each on the overall output of *n*’s own activation function.

- A **gradient** is a vector containing all the partial derivatives of a function with multiple variables. It essentially represents all the factors affecting the rate at which the output of a complex equation will change following a change in the input.

- The **chain rule** is a formula for calculating the derivatives of functions that involve not just multiple variables, but multiple functions. For example, consider a composite function *ƒ*(*x*) *= A*(*B(x*)). The derivative of the composite function, *f*, is equal to the derivative of the outer function (*A*) multiplied by the derivative of the inner function (*B*).



On a technical, mathematical level, the goal of backpropagation is to calculate the gradient of the loss function with respect to each of the individual parameters of the neural network. In simpler terms, backpropagation uses the chain rule to calculate the rate at which loss changes in response to any change to a specific weight (or bias) in the network.

Generally speaking, training neural networks with backpropagation entails the following steps:

- **A \*forward pass\****,* **making predictions on training data.**
- **A \*loss function\* measures the error of the model’s predictions during that forward pass.**
- ***Backpropagation\* of error, or a \*backward pass,\* to calculate the partial derivatives of the loss function.**
- ***Gradient descent,\* to update model weights.**





#### Activation functions

Consider a hypothetical hidden unit *z,* with a *tanh* activation function and bias term *t,* in the second layer of a neural network with 3 input nodes, *a*, *b* and *c,* in its input layer. Each of the connections between the input nodes and node *z* has a unique weight, *w.* We can describe the output value that node *z* will pass to the neurons in the next layer with the simplified equation *z* = *tanh*(*waz\*a + wbz\*b* + *wcz\*c* *+ t*)*.*

### Gradient descent



## convolution

https://jlebar.com/2023/9/11/convolutions.html
