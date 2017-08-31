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


def hardlim(x):
    return 1 if x > 0 else -1


class Perceptron(object):

    def __init__(self, arrf_file_path, learning_rate=0.5, training_set_portion=1.0,
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

        arff_file = arff.loads(open(arrf_file_path, "r"))
        inputs_qty = len(arff_file["attributes"])
        self._initialize_weights(inputs_qty)

        entries = arff_file["data"]
        index = ceil(len(entries) * training_set_portion)
        self._train_perceptron(entries[:index], learning_rate, max_iterations)

    # ##############
    # # VALIDATORS #
    # ##############
    def _validates_training_set_portion(self, training_set_portion):
        """Validates the portion of training set data
            Args:
                training_set_portion:  A float between 0 and 1.
            Raises:
                DatasetError: If `training_set_portion` is not between 0 and 1.
        """
        if not 0 < training_set_portion <= 1:
            raise PerceptronError(TRAINING_SET_PORTION_ERROR)

    # ###################
    # # PRIVATE METHODS #
    # ###################

    def _initialize_weights(self, inputs_qty):
        """Method which initializes all perceptron's weights randomly.
            Args:
               inputs_qty: The quantity of inputs a perceptron have.
            Returns:
               None
        """
        self.weights = np.array([random() for _ in range(inputs_qty)])

    def _train_perceptron(self, entries, learning_rate, max_iterations):
        """Method which trains the perceptron.
            Args:
                entries: Entries to train the perceptron.
                learning_rate: A float which is the learning_rate.
                max_iterations: The maximun number of iterations during training.
            Returns:
                None
        """
        for _ in range(max_iterations):

            adjusted_weights = False

            for entry in entries:
                inputs, desired_output = np.array([self.bias] + entry[:-1]), entry[:-1]
                summatory = sum(inputs * self.weights)
                output = hardlim(summatory)

                if desired_output != output:
                    self.weights += learning_rate * (desired_output - output) * inputs
                    adjusted_weights = True

            if not adjusted_weights:
                break

    # ###################
    # # PUBLIC METHODS #
    # ###################

    def classify_entry(self, inputs):
        """Method to classify an entry.
        Args:
            inputs: It is a list tha contains the inputs values.
        Returns:
            It returns an integer.
        """
        inputs = np.array([self.bias] + inputs)
        summatory = sum(inputs * self.weights)

        return hardlim(summatory)


