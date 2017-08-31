# -*- coding: utf-8 -*-

# Copyright (C) 2017-2017
# Author: Douglas Henrique Santana da Silva <douglas_hss@hotmail.com.br>
#
# For license information, see LICENSE.md


import arff
import numpy as np

from math import ceil
from random import random

from CustomExceptions import PerceptronError


TRAINING_SET_PORTION_ERROR = "The argument `training_set_portion` should be a float" + \
                             " between 0 and 1."
PERCEPTRON_IS_NOT_TRAINED = "Before classifying the perceptron should be trained."


def hardlim(x):
    return 1 if x > 0 else -1


class Perceptron(object):

    def __init__(self, arff_file_path, learning_rate=0.5, training_set_portion=1.0,
                 max_iterations=3):
        """Class contructor
            Args:
                arrf_file_path: String of the file path.
                learning_rate: A float which is the learning rate.
                training_set_portion: A float which identifies the portion of training set data.
                max_iterations: The maximun number of iterations during training.
        """
        self._validates_training_set_portion(training_set_portion)

        self.bias = -1
        self.learning_rate = learning_rate
        self.training_set_portion = training_set_portion
        self.max_iterations = max_iterations
        self.arff_file = arff.loads(open(arff_file_path, "r"))
        self.is_trained = False

        self._initialize_weights()

    # ##############
    # # VALIDATORS #
    # ##############
    def _validates_training_set_portion(self, training_set_portion):
        """Validates the portion of training set data
            Args:
                training_set_portion:  A float between 0 and 1.
            Raises:
                PerceptronError: If `training_set_portion` is not between 0 and 1.
        """
        if not 0 < training_set_portion <= 1:
            raise PerceptronError(TRAINING_SET_PORTION_ERROR)

    # ###################
    # # PRIVATE METHODS #
    # ###################

    def _initialize_weights(self):
        """Method which initializes all perceptron's weights randomly."""
        inputs_qty = len(self.arff_file["attributes"])
        self.weights = np.array([random() for _ in range(inputs_qty)])

    # ###################
    # # PUBLIC METHODS #
    # ###################

    def train(self):
        """Method which trains the perceptron."""

        entries = self.arff_file["data"]
        index = ceil(len(entries) * self.training_set_portion)
        entries = entries[:index]

        for _ in range(self.max_iterations):
            adjusted_weights = False

            for entry in entries:
                inputs, desired_output = np.array([self.bias] + entry[:-1]), float(entry[-1])
                summatory = sum(inputs * self.weights)
                output = hardlim(summatory)

                if desired_output != output:
                    self.weights += self.learning_rate * (desired_output - output) * inputs
                    adjusted_weights = True

            if not adjusted_weights:
                break

        self.is_trained = True

    def classify(self, inputs):
        """Method to classify an entry.
        Args:
            inputs: It is a list tha contains the inputs values.
        Returns:
            It returns an integer.
        Raises:
                PerceptronError: If perceptron is not trained.
        """
        if self.is_trained:
            inputs = np.array([self.bias] + inputs)
            summatory = sum(inputs * self.weights)

            return hardlim(summatory)
        else:
            raise PerceptronError(PERCEPTRON_IS_NOT_TRAINED)
