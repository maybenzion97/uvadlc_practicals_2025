################################################################################
# MIT License
#
# Copyright (c) 2025 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2025
# Date Created: 2025-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Kaiming initialization for weights
        if input_layer:  # for input layer: std = sqrt(1/in_features)
            std = np.sqrt(1.0 / in_features)
        else:  # for hidden layers (with ELU): std = sqrt(2/in_features)
            std = np.sqrt(2.0 / in_features)

        self.params['weight'] = np.random.normal(0, std, (out_features, in_features))
        self.params['bias'] = np.zeros(out_features)

        # initialize gradients with zeros
        self.grads['weight'] = np.zeros((out_features, in_features))
        self.grads['bias'] = np.zeros(out_features)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # store the input so we will be able to use it in the backward pass
        self.x = x

        # linear layer formula: Y = X @ W.T + B
        out = np.matmul(x, self.params['weight'].T) + self.params['bias']

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # gradient with respect to input: dx = dout @ W
        dx = np.matmul(dout, self.params['weight'])

        # gradient with respect to weights: dW = dout.T @ x
        self.grads['weight'] = np.matmul(dout.T, self.x)

        # gradient with respect to bias: db = sum(dout, axis=0)
        self.grads['bias'] = np.sum(dout, axis=0)

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # store the input so we will be able to use it in the backward pass
        self.x = x

        # ELU activation:
        #     f(x) = x                      , if x > 0,
        #     f(x) = alpha * (exp(x) - 1)   , if x <= 0
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # ELU derivative:
        #     f'(x) = 1                , if x > 0
        #     f'(x) = alpha * exp(x)   , if x <= 0
        # apply chain rule: dx = dout * f'(x)
        dx = np.where(self.x > 0, dout, dout * self.alpha * np.exp(self.x))

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Max Trick for numerical stability: subtract max from each row
        # softmax(x) with max trick = exp(x - max(x)) / sum(exp(x - max(x)))
        x_max = np.max(x, axis=1, keepdims=True)
        x_shifted = x - x_max

        # Compute exponentials
        exp_x = np.exp(x_shifted)

        # Compute sum of exponentials for each sample
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)

        # Compute softmax probabilities
        out = exp_x / sum_exp

        # Store for backward pass
        self.out = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # the Jacobian of softmax is: ∂s_i/∂x_j = s_i * (δ_ij - s_j)
        # where δ_ij = 1 if i==j, else 0

        # using chain rule: dx_j = Σ_i (dL/ds_i * ∂s_i/∂x_j)
        #                        = Σ_i (dout_i * s_i * (δ_ij - s_j))
        #                        = Σ_i (dout_i * s_i * δ_ij) - Σ_i (dout_i * s_i * s_j)
        #                        = s_j * dout_j - s_j * Σ_i(dout_i * s_i)
        #                        = s_j * (dout_j - Σ_i(dout_i * s_i))

        # compute (dout_i * s_i) for all i
        element_product = dout * self.out  # Shape: (batch_size, num_classes)

        # compute Σ_i(dout_i * s_i) for each sample
        sum_term = np.sum(element_product, axis=1, keepdims=True)  # Shape: (batch_size, 1)

        # compute (dout_j - Σ_i(dout_i * s_i)) for each j
        diff = dout - sum_term  # Shape: (batch_size, num_classes)

        # compute s_j * (dout_j - Σ_i(dout_i * s_i))
        dx = self.out * diff  # Shape: (batch_size, num_classes)

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.out = None

        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Cross Entropy Loss formula: L = -1/N * Σ(y_i * log(x_i)) where:
        #          x_i is the predicted probability for class i
        #          y_i is 1 for the correct class, 0 for others (one-hot encoded)
        # this simplifies to: L = -1/N * Σ log(x[correct_class])

        batch_size = x.shape[0]
        epsilon = 1e-10  # to prevent log(0)

        # compute: -1/N * Σ log(x[i, y[i]])
        out = -np.mean(np.log(x[np.arange(batch_size), y] + epsilon))

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Cross-entropy loss: L = -1/N * Σ(y_i * log(x_i))

        # taking derivative with respect to x_j:
        #   ∂L/∂x_j = -1/N * y_j / x_j
        # therefore: dx = -y / x / N

        batch_size = x.shape[0]

        # convert class indices to one-hot encoding
        y_one_hot = np.zeros_like(x)
        y_one_hot[np.arange(batch_size), y] = 1

        # compute dx = -y / x / N
        epsilon = 1e-10  # add epsilon to prevent division by zero
        dx = -y_one_hot / (x + epsilon) / batch_size

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx