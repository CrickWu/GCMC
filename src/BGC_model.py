import random
import cPickle
import lasagne
from theano import sparse
import lasagne.layers as L
import theano.tensor as T
from theano.tensor.slinalg import kron
import theano
import layers
import numpy as np
import scipy.sparse as sp

try:
    from pypropack import svdp # new svd package from https://github.com/jakevdp/pypropack
except ImportError:
    from scipy.sparse.linalg import svds as svdp

from BGC_base_model import BGC_base_model


class BGC_model(BGC_base_model):
    """ Bipartite Graph Convolution model
    """
    def __init__(self, lr, hidden_list, seed, model_file):

        super(BGC_model, self).__init__(model_file)

        self.g_hidden_list = hidden_list
        self.h_hidden_list = hidden_list

        self.learning_rate = lr
        lasagne.random.set_rng(np.random)
        np.random.seed(seed)
        random.seed(seed)

    def add_data(self, g, h, trn_graph, trn_x_index, trn_y_index, tst_graph, tst_x_index, tst_y_index, k=500,
                 pos_up_ratio=5.0):
        """
        """
        self.g = g                 # ng * ng
        self.h = h                 # nh * nh
        self.trn_graph = trn_graph # ng * nh (data are the corresponding instances)
        self.tst_graph = tst_graph # ng * nh (data are the corresponding instances)
        self.ng = g.shape[0]
        self.nh = h.shape[0]
        self.sym_g = self.gen_sym_graph(self.g)
        self.sym_h = self.gen_sym_graph(self.h)
        U, s, Vh = svdp(self.trn_graph, k=k)
        self.gX = U * np.sqrt(s)
        self.hX = Vh.T * np.sqrt(s)

        self.pos_trn_x_index, self.pos_trn_y_index = self.trn_graph.nonzero()
        self.trn_x_index, self.trn_y_index = trn_x_index, trn_y_index
        self.tst_x_index, self.tst_y_index = tst_x_index, tst_y_index
        self.pos_up_ratio = pos_up_ratio

        print 'bipartite shape:', trn_graph.shape
        print 'pos_num:', len(self.pos_trn_x_index)
        print 'total training:', len(self.trn_x_index)
        print 'pos_up_ratio:', self.pos_up_ratio


    def gen_sym_graph(self, A_ori):
        A = (A_ori + A_ori.transpose()) / 2.0  # changed to float64
        A = sp.csr_matrix(A, dtype='float32')
        A.setdiag(1.0)        # A_tilde = A + I_n
        D = 1.0 / np.sqrt(np.array(A.sum(axis=1)).reshape(-1,))
        D_inv_one_half = sp.diags(D, offsets=0)
        return D_inv_one_half.dot(A).dot(D_inv_one_half)

    def build_one_side(self, X, A, x, a, hidden_list):
        """
        :param X: theano param  # N times F
        :param A: theano param  # N times N
        :param x: real x, for determining the dimension
        :param a: real a, for determining the dimension
        :return:
        """
        l_x_in = lasagne.layers.InputLayer(shape=(a.shape[0], x.shape[1]), input_var=X) 

        cur_layer = layers.DenseGraphCovLayer(l_x_in, A, hidden_list[0], nonlinearity=lasagne.nonlinearities.tanh)

        for hidden_unit in hidden_list[1:]:
            cur_layer = layers.DenseGraphCovLayer(cur_layer, A, hidden_unit, nonlinearity=lasagne.nonlinearities.tanh)
        return lasagne.layers.get_output(cur_layer), cur_layer


    def build(self, pre_load=False, binary_graph=True):
        """build the model. This method should be called after self.add_data.
        """
        hA = sparse.csr_matrix('hA', dtype='float32')   # nh times nh
        gA = sparse.csr_matrix('gA', dtype='float32')   # ng times ng
        Y = sparse.csr_matrix('Y', dtype='float32')     # ng times nh
        x_index = T.ivector('xind')                     #
        y_index = T.ivector('yind')                     #

        # not sparse (due to SVD)
        hX = T.fmatrix('hX')   # nh times Fh
        gX = T.fmatrix('gX')   # ng times Fg

        # final dimension equals
        assert self.g_hidden_list[-1] == self.h_hidden_list[-1]
        g_pred, g_net = self.build_one_side(gX, gA, self.gX, self.sym_g, self.g_hidden_list)
        h_pred, h_net = self.build_one_side(hX, hA, self.hX, self.sym_h, self.h_hidden_list)

        # final layer g_pred * h_pred^T
        Y_pred = T.dot(g_pred, h_pred.T) # ng times nh

        # squared matrix
        loss_mat = lasagne.objectives.squared_error(Y_pred, Y)
        if binary_graph:
            loss = (loss_mat[x_index, y_index].sum()
            + loss_mat[self.pos_trn_x_index, self.pos_trn_y_index].sum() * self.pos_up_ratio) \
            / (x_index.shape[0] + self.pos_trn_x_index.shape[0])
        else:
            loss = loss_mat[x_index, y_index].mean()

        g_params = lasagne.layers.get_all_params(g_net)
        h_params = lasagne.layers.get_all_params(h_net)
        params = g_params + h_params
        self.l = [g_net, h_net]

        updates = lasagne.updates.adam(loss, params)


        grads = lasagne.updates.get_or_compute_grads(loss, params)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))

        self.train_fn = theano.function([gX, hX, gA, hA, Y, x_index, y_index], [Y_pred, loss, grad_norm],
                                        updates=updates, on_unused_input='ignore', allow_input_downcast=True)
        self.test_fn = theano.function([gX, hX, gA, hA], Y_pred,
                                       on_unused_input='ignore', allow_input_downcast=True)
        # loading the parameters
        if pre_load:
            self.load_params()


    def step_train(self, max_iter):
        """a training step. Iteratively sample batches for three loss functions.
        max_iter (int): # iterations for the current training step.
        """
        nzx, nzy = self.trn_graph.nonzero()
        n = len(self.trn_x_index)
        n_pos = len(nzx)
        for _ in range(max_iter):
            Y_pred, loss, grad_norm = self.train_fn(self.gX, self.hX, self.sym_g, self.sym_h,
                                                    self.trn_graph, self.trn_x_index, self.trn_y_index)
        return Y_pred, loss, grad_norm

    def predict(self):
        """predict the dev or test instances.
        """
        return self.test_fn(self.gX, self.hX, self.sym_g, self.sym_h)

