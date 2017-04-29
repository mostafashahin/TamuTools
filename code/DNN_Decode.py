"""
"""
from __future__ import print_function
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
#Parameters
DataSet = sys.argv[1]
n_hls = int(sys.argv[2])
n_hus = int(sys.argv[3])
n_out = int(sys.argv[4])
batchSize = int(sys.argv[5])
n_inp = int(sys.argv[6])
outFile = sys.argv[7]
paramFile = sys.argv[8]
#print 'Running with parameters:\nDataSet = {0:10}\nLearning rate = {1:1.7f}\nNumber of Hidden layers = {2:2d}\nNumber of hidden units = {3:4d}\nBatch size = {4:4d}'.format(DataSet,lr,n_hls,n_hus,batchSize)



class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=n_inp,
                 hidden_layers_sizes=[50], n_outs=n_out):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
                                 # of [int] labels

        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        #activation=T.nnet.sigmoid)
                                        activation=T.tanh)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        self.y_pred = self.logLayer.y_pred
        self.p_y = self.logLayer.p_y_given_x

    def build_finetune_functions(self, datasets, batch_size):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (test_set_x, test_set_y) = datasets

        # compute number of minibatches for training, validation and testing
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches = (n_test_batches/batch_size) + (n_test_batches % batch_size > 0)

        print (n_test_batches)

        index = T.lscalar('index')  # index to a [mini]batch
        Alr = T.scalar('Alr')  # learning rate to use

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * Alr))

        test_score_i = theano.function([index], [self.errors, self.y_pred,self.p_y],
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]})

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i)[0] for i in xrange(n_test_batches)]
        def yPred():
            yP = numpy.asarray([])
            for i in xrange(n_test_batches):
                yP = numpy.concatenate((yP,test_score_i(i)[1]))
            return yP
        def Py_given_x():
            Py = numpy.empty((1,n_out))
            for i in xrange(n_test_batches):
                Py = numpy.concatenate((Py,test_score_i(i)[2]))
            Py = numpy.delete(Py,0,axis=0)
            return Py
        return test_score, yPred, Py_given_x


def test_DBN(dataset=DataSet, batch_size=batchSize):
    """
    Demonstrates how to train and test a Deep Belief Network.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining
    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    :type k: int
    :param k: number of Gibbs steps in CD/PCD
    :type training_epochs: int
    :param training_epochs: maximal number of iterations ot run the optimizer
    :type dataset: string
    :param dataset: path the the pickled dataset
    :type batch_size: int
    :param batch_size: the size of a minibatch
    """
    f = gzip.open(dataset)
    
    data = cPickle.load(f)#load_data(dataset)
    f.close()
    test_set_x, test_set_y = data
    shared_x = theano.shared(numpy.asarray(test_set_x, dtype=theano.config.floatX), borrow=True)
    shared_y = theano.shared(numpy.asarray(test_set_y, dtype=theano.config.floatX), borrow=True)
    datasets = (shared_x,T.cast(shared_y, 'int32'))
    # compute number of minibatches for training, validation and testing
    #n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print ('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_inp,
              hidden_layers_sizes=[n_hus for i in range(n_hls)],
              n_outs=n_out)
    #Load params
    fbestParam = gzip.open(paramFile,'rb')
    best_params = cPickle.load(fbestParam)
    fbestParam.close()
    for param, best_param in zip(dbn.params,best_params):
	print(type(best_param))
        param.set_value(numpy.float32(best_param))
    for param in dbn.params:
        print(param.get_value().shape)
    test_model, gety_pred, getP_y = dbn.build_finetune_functions(
                datasets=datasets, batch_size=batch_size)

    test_losses = test_model()
    test_score = numpy.mean(test_losses)
    Py = getP_y()
    print (Py.shape)
    numpy.savetxt('p_y_x',Py)
    yP = gety_pred()
    numpy.savetxt('y_pre.txt',yP)
    y = test_set_y#.owner.inputs[0].get_value()
    print (yP.shape)
    print (y.shape)
    print (yP)
    print (accuracy_score(y,yP))
    print (confusion_matrix(y,yP))
    print (classification_report(y,yP))
 
if __name__ == '__main__':
    test_DBN()
