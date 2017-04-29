"""
"""
from __future__ import print_function
import cPickle
import gzip
import os
import sys
import time
import math
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from logistic_sgd_joblib import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM
from GetDataFromList import GetXy
if len(sys.argv) != 10:
    print ('Usage: DBN_TrainFromFiles.py learning_rate N#HiddenLayers N#HiddenUnits N#Outputs BatchSize N#Epochs OutFile AttributeIndx NumFrames')
    sys.exit(1)
#Parameters
lr = float(sys.argv[1])
n_hls = int(sys.argv[2])
n_hus = int(sys.argv[3])
n_out = int(sys.argv[4])
batchSize = int(sys.argv[5])
NOfepoch = int(sys.argv[6])
outFile = sys.argv[7]
iAttributeIndx = int(sys.argv[8])
iNumFrames = int(sys.argv[9])
sTrainingList = 'TrainingFeatMask.scp'
sValidList = 'ValidFeatMask.scp'
sTestingList = 'TestingFeatMask.scp'
#testFile = sys.argv[10]
#iAttributeIndx = 3 
iFeatSize = 78
iNumFiles = 300
n_inp = iFeatSize * iNumFrames
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
    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index,
                            theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={self.x:
                                    train_set_x[batch_begin:batch_end]})
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, sValidFeatMask, sTestFeatMask, batch_size, learning_rate):
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

        #(train_set_x, train_set_y) = datasets[0]
        #(valid_set_x, valid_set_y) = datasets[1]
        #(test_set_x, test_set_y) = datasets[2]
        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
            shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype='int32'),#theano.config.floatX),
                             borrow=borrow)
            return shared_x, shared_y#T.cast(shared_y, 'int32')


        with open(sValidFeatMask,'r') as fFeatMask:
            lLines = fFeatMask.read().splitlines()
            lValidFeatMaskFiles = [tuple(line.split()) for line in lLines]

        with open(sTestFeatMask,'r') as fFeatMask:
            lLines = fFeatMask.read().splitlines()
            lTestFeatMaskFiles = [tuple(line.split()) for line in lLines]
        print('Loading Validation Data.....')
        valid_set_x,valid_set_y = shared_dataset(GetXy(lValidFeatMaskFiles,iAttributeIndx,iNumFrames,iFeatSize,True))
        print('Done')
        print('Loading Testing Data.....')
        test_set_x,test_set_y = shared_dataset(GetXy(lTestFeatMaskFiles,iAttributeIndx,iNumFrames,iFeatSize,True))
        print('Done')
 
        # compute number of minibatches for training, validation and testing
        n_valid_samples = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = n_valid_samples/ batch_size
        n_test_samples = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches = (n_test_samples/batch_size) + (n_test_samples % batch_size > 0)

        print (('Number of Validation Samples = %i , Number of Validation batches = %i , Number of Test Samples = %i , Number of Test batches = %i')%(n_valid_samples,n_valid_batches,n_test_samples,n_test_batches))

        index = T.lscalar('index')  # index to a [mini]batch
        Alr = T.scalar('Alr')  # learning rate to use
        X = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * Alr))

        train_fn = theano.function(inputs=[X,y, Alr],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: X,
                      self.y: y})

        test_score_i = theano.function([index], [self.errors, self.y_pred],
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})
        #yPred = theano.function([self.x],self.y_pred)
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i)[0] for i in xrange(n_test_batches)]
        def yPred():
            yP = numpy.asarray([])
            for i in xrange(n_test_batches):
                yP = numpy.concatenate((yP,test_score_i(i)[1]))
            return yP
        def GetRefY():
            return test_set_y.get_value(borrow=True)
        return train_fn, valid_score, test_score, yPred, GetRefY

def test_DBN(finetune_lr=lr, pretraining_epochs=100,
             pretrain_lr=0.0025, k=2, training_epochs=NOfepoch,
             sValidList=sValidList, sTestingList=sTestingList, batch_size=batchSize):
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
    # compute number of minibatches for training, validation and testing
    with open(sTrainingList,'r') as fFeatMask:
        lLines = fFeatMask.read().splitlines()
        lFeatMaskFiles = [tuple(line.split()) for line in lLines]
    n_train_chuncks = int(math.ceil(len(lFeatMaskFiles)/float(iNumFiles)))#train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    # numpy random generator
    numpy_rng = numpy.random.RandomState(123)
    print ('... building the model')
    # construct the Deep Belief Network
    dbn = DBN(numpy_rng=numpy_rng, n_ins=n_inp,
              hidden_layers_sizes=[n_hus for i in range(n_hls)],
              n_outs=n_out)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    #print ('... getting the pretraining functions')
    #pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
    #                                            batch_size=batch_size,
    #                                            k=k)

    #print ('... pre-training the model')
    #start_time = time.clock()
    ## Pre-train layer-wise
    #for i in xrange(dbn.n_layers):
    #    # go through pretraining epochs
    #    for epoch in xrange(pretraining_epochs):
    #        # go through the training set
    #        c = []
    #        for batch_index in xrange(n_train_batches):
    #            c.append(pretraining_fns[i](index=batch_index,
    #                                        lr=pretrain_lr))
    #        print ('Pre-training layer %i, epoch %d, cost ' % (i, epoch))
    #        print (numpy.mean(c))
#
 #   end_time = time.clock()
 #   print (('The pretraining code for file ' +
 #                         os.path.split(__file__)[1] +
 #                         ' ran for %.2fm' % ((end_time - start_time) / 60.)))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print ('... getting the finetuning functions')
    train_fn, validate_model, test_model, gety_pred, get_y = dbn.build_finetune_functions(
                sValidFeatMask=sValidList, sTestFeatMask=sTestingList, batch_size=batch_size,
                learning_rate=finetune_lr)

    print ('... finetunning the model')
    # early-stopping parameters
    #patience = 4 * n_train_batches  # look as this many examples regardless
    #patience_increase = 2.    # wait this much longer when a new best is
                              # found
    #improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    #validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    print('Number of Training Chunks = ',n_train_chuncks)
    y = get_y()
    best_params = []
    ybest = numpy.asarray([])
    best_validation_loss = numpy.inf
    prev_validation_loss = 200
    test_score = 0.
    start_time = time.clock()
    Alrc = 0.1
    AlrE = 0.00001 
    done_looping = False
    epoch = 0
    epochC = 0
    iter = 0 
    for param in dbn.params:
        best_params.append(param.get_value())

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        epochC = epochC + 1
        #for minibatch_index in xrange(n_train_batches):
        for chunck_index in xrange(n_train_chuncks):
            #print n_train_batches, epoch, minibatch_index
            #minibatch_avg_cost = train_fn(minibatch_index,Alrc)
            lCurFeatMask = lFeatMaskFiles[chunck_index*iNumFiles:(chunck_index+1)*iNumFiles]
            arCur_X,vCur_y = GetXy(lCurFeatMask,iAttributeIndx,iNumFrames,iFeatSize,True)
            print(('Chunk Index = %i , Number of Samples = %i')%(chunck_index,arCur_X.shape[0]))
            #minibatch_avg_cost = train_fn(train_set_x_NS[minibatch_index*batch_size:(minibatch_index+1)*batch_size],train_set_y_NS[minibatch_index*batch_size:(minibatch_index+1)*batch_size],Alrc)
            n_curTrain_batches = int(math.ceil(arCur_X.shape[0]/float(batch_size)))
            for minibatch_index in xrange(n_curTrain_batches):
                minibatch_avg_cost = train_fn(arCur_X[minibatch_index*batch_size:(minibatch_index+1)*batch_size],vCur_y[minibatch_index*batch_size:(minibatch_index+1)*batch_size],Alrc)
                #iter = (epoch - 1) * n_train_chuncks + chunck_index * minibatch_index
        validation_losses = validate_model()                
        this_validation_loss = numpy.mean(validation_losses)
        lossratio = (this_validation_loss - prev_validation_loss)/(prev_validation_loss+1)
        print (lossratio)
        print('epoch %i, validation error %f, lr %f %%' % \
             (epoch,this_validation_loss * 100., Alrc))

        # if we got the best validation score until now
        #if this_validation_loss < best_validation_loss:
        if lossratio <= 0.0:
            #print '*******************1**************'
            #print dbn.params[0].get_value()
            for i in range(len(dbn.params)):
                best_params[i] = dbn.params[i].get_value()

            # save best validation score and iteration number
            best_validation_loss = this_validation_loss
            prev_validation_loss = this_validation_loss
            best_iter = iter

            # test it on the test set
            test_losses = test_model()
            test_score = numpy.mean(test_losses)
            yP = gety_pred()
            acc = accuracy_score(y,yP)
            conf_matrix = confusion_matrix(y,yP)
            conf_matrix_norm = conf_matrix/numpy.sum(conf_matrix,axis=1,dtype='float',keepdims=True)
            acc_str = ' '.join([str(conf_matrix_norm[i,i]) for i in range(conf_matrix_norm.shape[0])])
            conf_str = ' '.join(conf_matrix.flatten().astype('str'))
            ybest=yP
            print(('     epoch %i, test error of best model %f, acc = %s , conf = %s %%') %
                  (epoch,test_score * 100., acc_str, conf_str))
        else:
            if Alrc <= AlrE:
                done_looping = True
                break
            elif epochC > 20:
                Alrc = Alrc/2
                for param, best_param in zip(dbn.params,best_params):
                    param.set_value(best_param)
                epochC = 0
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    if not os.path.isdir('Temp'):
        os.makedirs('Temp')
    fbestParam = gzip.open('Temp/Bestparam_'+str(iAttributeIndx)+'_'+str(iNumFrames)+'_'+str(n_hls)+'_'+str(n_hus)+'_'+str(n_out)+'.pkl.gz','wb')
    fbestP = open('Temp/Bestp_'+str(iAttributeIndx)+'_'+str(iNumFrames)+'_'+str(n_hls)+'_'+str(n_hus)+'_'+str(n_out)+'.txt','w')
    numpy.savetxt(fbestP,ybest)
    fbestP.close()
    cPickle.dump(best_params,fbestParam)
    fbestParam.close()
    OF = open(outFile,'a')
    print(iAttributeIndx,iNumFrames,lr ,n_hls, n_hus, n_out, batchSize, NOfepoch, n_inp, outFile, test_score * 100., acc_str, conf_str, file = OF)

    OF.close()
if __name__ == '__main__':
    test_DBN()
