import lasagne
from theano import sparse
import theano.tensor as T

class DenseGraphCovLayer(lasagne.layers.Layer):
    def __init__(self, H, A, num_units, W = lasagne.init.GlorotUniform(),
                 nonlinearity = lasagne.nonlinearities.rectify, **kwargs):
        super(DenseGraphCovLayer, self).__init__(H, **kwargs)
        num_inputs = self.input_shape[1]
        self.num_units = num_units
        self.W = self.add_param(W, (num_inputs, num_units), name="W", trainable=True)
        self.A = A
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        #Z = T.dot(self.A, input)
        Z = sparse.basic.structured_dot(self.A, input)
        activation = T.dot(Z, self.W)
        if self.nonlinearity != lasagne.nonlinearities.softmax:
            return self.nonlinearity(activation)
        else:
            return T.exp(activation) / (T.exp(activation).sum(1, keepdims = True))
