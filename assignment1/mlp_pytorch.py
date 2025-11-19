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
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes, use_batch_norm=False):
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
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        TODO:
        Implement module setup of the network.
        The linear layer have to initialized according to the Kaiming initialization.
        Add the Batch-Normalization _only_ is use_batch_norm is True.
        
        Hint: No softmax layer is needed here. Look at the CrossEntropyLoss module for loss calculation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        super(MLP, self).__init__()

        # build the network using nn.Sequential with OrderedDict for named layers
        layers = OrderedDict()
        current_input_size = n_inputs

        # add hidden layers with ELU activations (and optional batch norm)
        for idx, hidden_size in enumerate(n_hidden):
            # add linear layer
            layer_name = f'linear_{idx}'
            linear = nn.Linear(current_input_size, hidden_size)

            if idx == 0:
                nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='linear')
            else:
                nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(linear.bias)

            layers[layer_name] = linear

            # add batch normalization if requested
            if use_batch_norm:
                layers[f'batchnorm_{idx}'] = nn.BatchNorm1d(hidden_size)

            # add ELU activation
            layers[f'elu_{idx}'] = nn.ELU()

            # update input size for next layer
            current_input_size = hidden_size

        # add final output layer (Linear only, no activation)
        output_linear = nn.Linear(current_input_size, n_classes)

        if len(n_hidden) == 0:  # if no hidden layers, this is the first layer after input
            nn.init.kaiming_normal_(output_linear.weight, mode='fan_in', nonlinearity='linear')
        else:
            nn.init.kaiming_normal_(output_linear.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.zeros_(output_linear.bias)

        layers['output'] = output_linear

        # create sequential model
        self.model = nn.Sequential(layers)

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

        out = self.model(x)

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
    
