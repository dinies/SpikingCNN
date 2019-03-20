from abc import ABC,abstractmethod
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np
import math
from numba import jit
import itertools

class Layer( ABC):

    def __init__(self, padding_type, stride, expected_output_dim ):
        self.padding_type = padding_type
        self.stride = stride
        self.expected_output_dim = expected_output_dim

    @abstractmethod
    def makeOperation( self, input_to_layer):
        pass

    @abstractmethod
    def resetLayer( self):
        pass

    @abstractmethod
    def createWeights( self):
        pass

    @abstractmethod
    def saveWeights( self, path, index):
        pass

    @abstractmethod
    def loadWeights( self, path, index):
        pass

    @abstractmethod
    def getSynapseChangeInfo(self):
        pass

    @abstractmethod
    def getWeightsStatistics( self):
        pass



