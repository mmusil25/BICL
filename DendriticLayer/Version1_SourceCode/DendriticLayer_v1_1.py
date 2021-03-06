"""
Mark Musil
Undergraduate Honor's Thesis
November 2018

This file defines a Keras layer that implements a dendritically inspired
fully connected layer. Each neuron has several 'dendrites' that have their
own summation and activation functions that are evaluated prior to the
standard sum and sigmoid output.

For a complete definition and explanation of the layer see the document
"HonorsThesisProspectus.pdf" on my github in the Dendritic Layer folder
of the CorGraph Project repository.

CHANGE LOG:

Version 1.1 is the original, non-working version of the algorithm.


"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from math import exp, log

# import copy
# import types as python_types
# import warnings
#
# from keras import backend as K
# from keras.engine.topology import Layer

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer


# from keras.utils.generic_utils import func_dump
# from keras.utils.generic_utils import func_load
# from keras.utils.generic_utils import deserialize_keras_object
# from keras.utils.generic_utils import has_arg
# from keras.utils import conv_utils
# from keras.legacy import interfaces

def multi_variate_sigmoid(x):  # Here x is a vector
    return (1 + exp(np.sum(x))) ** (-1)


class Dendritic(Layer):
    """Just your regular densely-connected NN layer.

    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).

    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.

    # Example

    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)

        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```

    # Arguments
        units: Positive integer, dimensionality of the output space.
        dendrites: Dendritic banches per neuron
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
b            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.

    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    # @interfaces.legacy_dense_support
    def __init__(self, units, dendrites,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dendritic, self).__init__(**kwargs)
        self.units = units
        self.dendrites = dendrites
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.dendrites, self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.alpha_L = 0.5
        self.alpha_U = 0.5
        self.b_U = 0  # Might make these trainable
        self.b_L = 1  # Might make these trainable
        self.a_d = 1

        self.c_d = 0.5
        self.b_d = 0.5

        # self.b_d = self.add_weight(shape = 1,
        #                            initializer = self.bias_initializer,
        #                            name='b_d',
        #                            regularizer=self.bias_regularizer,
        #                            constraint=self.bias_constraint)
        # self.c_d = self.add_weight(shape = 1,
        #                            initializer = self.bias_initializer,
        #                            name='c_d',
        #                            regularizer=self.bias_regularizer,
        #                            constraint=self.bias_constraint)
        self.dendriteInput = np.zeros((self.dendrites, self.units))
        self.dendriteActivations = np.zeros(self.dendrites)
        self.preoutput = 0
        if self.use_bias:
            self.dendriteBias = self.add_weight(shape=(self.units * self.dendrites,),
                                                initializer=self.bias_initializer,
                                                name='bias',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)

            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim})
        #  self.built = Truef
        super(Dendritic, self).build(input_shape)

    def dendritic_boundary(self, x):  # Here x is a single valued real number
        numerator = (1 + exp(self.alpha_L * (x - self.b_L))) ** (self.alpha_L ** (-1))
        denominator = (1 + exp(self.alpha_U * (x - self.b_U))) ** (self.alpha_U ** (-1))
        return log(numerator / denominator) + self.b_L

    def dendritic_transfer(self, x):  # Here x is also vector
        arg1 = self.c_d * multi_variate_sigmoid(np.multiply(self.a_d, np.subtract(x, self.b_d)) + np.sum(x))
        return self.dendritic_boundary(arg1)

    def call(self, inputs):  # Layer logic
        for n in range(self.units):
            for d in range(self.dendrites):
                self.dendriteInput[d] = K.dot(inputs, self.kernel[d])
                if self.use_bias:
                    self.dendriteInput[d, :] = K.bias_add(self.dendriteInput[d, :], self.dendriteBias,
                                                          data_format='channels_last')
                self.dendriteActivations[d] = self.dendritic_transfer(self.dendriteInput[d])
                preoutput = np.sum(self.dendriteActivations)
                if self.use_bias:
                    output = K.bias_add(preoutput, self.bias, data_format='channels_last')
                if self.activation is not None:
                    output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dendritic, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
