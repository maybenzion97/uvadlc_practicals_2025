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
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, alpha=0.5):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
          alpha: ELU activation parameter

        TODO:
        Implement initialization of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.modules = []  # store all modules in order
        current_input_size = n_inputs

        # add hidden layers with ELU activations (only if n_hidden is not empty)
        for idx, hidden_size in enumerate(n_hidden):
            # add linear layer
            is_input_layer = (idx == 0)
            self.modules.append(LinearModule(current_input_size, hidden_size, input_layer=is_input_layer))
            # add ELU activation
            self.modules.append(ELUModule(alpha=alpha))
            # update input size for next layer
            current_input_size = hidden_size

        # add final output layer (Linear -> Softmax)
        is_input_layer = (len(n_hidden) == 0)
        self.modules.append(LinearModule(current_input_size, n_classes, input_layer=is_input_layer))

        # add softmax for multi-class classification (n_classes > 1)
        if n_classes > 1:
            self.modules.append(SoftMaxModule())

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # pass input through all modules sequentially
        out = x
        for module in self.modules:
            out = module.forward(out)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # propagate gradients backward through all modules in reverse order - start with dout and pass it through each module's backward
        for module in reversed(self.modules):
            dout = module.backward(dout)

        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # clear cache from all modules
        for module in self.modules:
            module.clear_cache()

        #######################
        # END OF YOUR CODE    #
        #######################
