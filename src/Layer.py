from abc import ABC

class Layer( ABC):

    def __init__(self, padding_type, stride, expected_output_dim ):
        self.padding_type = padding_type
        self.stride = stride
        self.expected_output_dim = expected_output_dim


class ConvolutionalLayer(Layer):

    def __init__(self, padding_type, stride, filter_dim, threshold_potential, expected_output_dim):
        super().__init__( padding_type, stride, expected_output_dim)
        self.filter_dim = filter_dim
        self.threshold_potential = threshold_potential


class PoolingLayer(Layer):

    def __init__(self,padding_type,stride, window_shape, pooling_type, expected_output_dim ):
        super().__init__( padding_type, stride, expected_output_dim)
        self.window_shape = window_shape
        self.pooling_type = pooling_type

