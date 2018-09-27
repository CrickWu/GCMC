import lasagne
import cPickle
import random
import numpy as np

class BGC_base_model(object):

    def __init__(self, model_file):
        self.model_file = model_file

    def store_params(self):
        """serialize the model parameters in self.model_file.
        """

        for i, l in enumerate(self.l):
            fout = open("{}.{}".format(self.model_file, i), 'w')
            params = lasagne.layers.get_all_param_values(l)
            cPickle.dump(params, fout, cPickle.HIGHEST_PROTOCOL)
            fout.close()

    def load_params(self):
        """load the model parameters from self.model_file.
        """
        for i, l in enumerate(self.l):
            fin = open("{}.{}".format(self.model_file, i))
            params = cPickle.load(fin)
            lasagne.layers.set_all_param_values(l, params)
            fin.close()
