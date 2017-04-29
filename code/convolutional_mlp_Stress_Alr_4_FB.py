"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

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
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


DataSet = sys.argv[1]
n_out = int(sys.argv[2])
fsx = int(sys.argv[3])
fsy = int(sys.argv[4])
p = int(sys.argv[5])
#p2 = int(sys.argv[4])
cls1 = int(sys.argv[6])
cls2 = int(sys.argv[7])
nhu1 = int(sys.argv[8])
nFB = int(sys.argv[9])
nFs = int(sys.argv[10])
iFMs = int(sys.argv[11])
nhus = int(sys.argv[12])
outFile = sys.argv[13]
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(p, p)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset=DataSet,
                    nkerns=[cls1, cls2], batch_size=100):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    print (type(train_set_x))

    #train_set_x.set_value(train_set_x.get_value(borrow=True)[:,:540])
    #valid_set_x.set_value(valid_set_x.get_value(borrow=True)[:,:540])
    #test_set_x.set_value(test_set_x.get_value(borrow=True)[:,:540])
    
    #train_set_x = train_set_x / 100
    #valid_set_x = valid_set_x / 100
    #test_set_x = test_set_x / 100
    

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    #n_test_batches = (n_test_batches/batch_size) + (n_test_batches % batch_size > 0)
  
    print (n_test_batches)
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    Alr = T.scalar('Alr')
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (nFB, nFs)  # this is the size of MNIST images

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print ('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    dFeatureV = iFMs*nFB*nFs
    xinp = x[:,:dFeatureV]
    
#    print (x.shahpe)
    
    layer0_input = xinp.reshape((batch_size, iFMs, nFB, nFs))
    layer1H_input = x[:,dFeatureV:]
    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, iFMs, nFB, nFs),
            filter_shape=(nkerns[0], iFMs, fsx, fsy), poolsize=(p, p))
    cl2x = (nFB - fsx + 1)/p
    cl2y = (nFs - fsy + 1)/p
    layer1H = HiddenLayer(rng, input = layer1H_input, n_in = 14, n_out = nhus, activation = T.tanh)
    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    
    #layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
    #        image_shape=(batch_size, nkerns[0], cl2x, cl2y),
    #        filter_shape=(nkerns[1], nkerns[0], fsx, 1), poolsize=(p2, 1))
    #hl1 = (cl2x - fsx + 1)/p2
    hl1 = cl2x * cl2y
    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer0.output.flatten(2)
    #layer2_inputT = T.concatenate([layer2_input,x[:,dFeatureV:]],axis = 1)
    layer2_inputT = T.concatenate([layer2_input,layer1H.output],axis = 1)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_inputT, n_in=(nkerns[0] * hl1 * 1)+nhus,
                         n_out=nhu1, activation=T.tanh)

    #layer22 = HiddenLayer(rng, input=layer2.output, n_in=nhu1,
    #                     n_out=nhu1, activation=T.tanh)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=nhu1, n_out=n_out)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)
    #yPred = layer3.ypred(layer2.output)
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], [layer3.errors(y), layer3.y_pred],
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    

    # create a list of all model parameters to be fit by gradient descent
    #params = layer3.params + layer22.params + layer2.params + layer1.params + layer0.params
    params = layer3.params + layer2.params + layer1H.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        #updates.append((param_i, param_i - learning_rate * grad_i))
        updates.append((param_i, param_i - Alr * grad_i))

    train_model = theano.function([index, Alr], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size][:],
            y: train_set_y[index * batch_size: (index + 1) * batch_size][:]})

    ###############
    # TRAIN MODEL #
    ###############
    print ('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    #best_params = None
    best_params = []
    best_validation_loss = numpy.inf
    prev_validation_loss = 200

    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    Alrc = 0.2
    AlrE = 0.00001
    epochC = 0 
    epoch = 0
    done_looping = False
    for param in params:
        best_params.append(param.get_value())
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        epochC = epochC + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print ('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index, Alrc)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                lossratio = (this_validation_loss - prev_validation_loss)/(prev_validation_loss+1)
                print (lossratio)
                print('epoch %i, minibatch %i/%i, validation error %f, lr %f %%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100., Alrc))

                # if we got the best validation score until now
                #if this_validation_loss < best_validation_loss:
                if lossratio <= 0.0:
                    for i in range(len(params)):
                        best_params[i] = params[i].get_value()
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    prev_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    #tm =  test_model(0)
                   
                    yP = numpy.asarray([])
                    test_losses = [test_model(i)[0] for i in xrange(n_test_batches)]
                    for i in xrange(n_test_batches):
                        yP = numpy.concatenate((yP,test_model(i)[1]))
                    print (yP.shape)
                    test_score = numpy.mean(test_losses)
                    
                    #yP = yPred#yPred(layer2.output.owner.inputs[0].get_value())
                    y = test_set_y.owner.inputs[0].get_value()[:3000]
                    
                    print (yP.shape)
                    print (y.shape)
                    I1 = numpy.nonzero(y==0.0)
                    I2 = numpy.nonzero(y==1.0)
                    I3 = numpy.nonzero(y==2.0)
                    I4 = numpy.nonzero(y==3.0)
                    print (I1[0].shape)
                    print (I2[0].shape)
                    print (I3[0].shape)
                    print (I4[0].shape)
                    I11 = numpy.nonzero(yP[I1[0]]==0)
                    I12 = numpy.nonzero(yP[I1[0]]==1)
                    I13 = numpy.nonzero(yP[I1[0]]==2)
                    I14 = numpy.nonzero(yP[I1[0]]==3)
                    I21 = numpy.nonzero(yP[I2[0]]==0)
                    I22 = numpy.nonzero(yP[I2[0]]==1)
                    I23 = numpy.nonzero(yP[I2[0]]==2)
                    I24 = numpy.nonzero(yP[I2[0]]==3)
                    I31 = numpy.nonzero(yP[I3[0]]==0)
                    I32 = numpy.nonzero(yP[I3[0]]==1)
                    I33 = numpy.nonzero(yP[I3[0]]==2)
                    I34 = numpy.nonzero(yP[I3[0]]==3)
                    I41 = numpy.nonzero(yP[I4[0]]==0)
                    I42 = numpy.nonzero(yP[I4[0]]==1)
                    I43 = numpy.nonzero(yP[I4[0]]==2)
                    I44 = numpy.nonzero(yP[I4[0]]==3)

                    acc1 = float(float(I11[0].size)/float(I1[0].size))
                    acc2 = float(float(I22[0].size)/float(I2[0].size))
                    if n_out == 3:
                    	acc3 = float(float(I33[0].size)/float(I3[0].size))
                    elif n_out == 4:
                    	acc3 = float(float(I33[0].size)/float(I3[0].size))
                        acc4 = float(float(I44[0].size)/float(I4[0].size))
                    else:
                        acc3 = 0
                        acc4 = 0
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f, acc1 = %f, acc2 = %f, acc3 = %f, acc4 = %f, I11 = %i, I12 = %i, I13 = %i, I14 = %i, I21 = %i, I22 = %i, I23 = %i, I24 = %i, I31 = %i, I32 = %i, I33 = %i, I34 = %i, I41 = %i, I42 = %i, I43 = %i, I44 = %i %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100., acc1 * 100., acc2 * 100., acc3 * 100, acc4 * 100, I11[0].size, I12[0].size, I13[0].size, I14[0].size, I21[0].size, I22[0].size, I23[0].size, I24[0].size, I31[0].size, I32[0].size, I33[0].size, I34[0].size, I41[0].size, I42[0].size, I43[0].size, I44[0].size))

                    #print(('     epoch %i, minibatch %i/%i, test error of best '
                    #       'model %f %%') %
                    #      (epoch, minibatch_index + 1, n_train_batches,
                    #       test_score * 100.))
                else:
                    if Alrc <= AlrE:
                        done_looping = True
                        break
                    elif epochC > 40:
                        Alrc = Alrc/2
                        for param, best_param in zip(params,best_params):
                            param.set_value(best_param)
                        epochC = 0
            #if patience <= iter:
            #    done_looping = True
            #    break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    #print >> sys.stderr, ('The code for file ' +
    #                      os.path.split(__file__)[1] +
    #                      ' ran for %.2fm' % ((end_time - start_time) / 60.))
    OF = open(outFile,'a')
    print(DataSet, n_out, fsx, fsy, p, cls1, cls2, nhu1, nFB, nFs, iFMs, nhus, batch_size, test_score * 100., acc1 * 100., acc2 * 100., acc3 * 100, acc4 * 100, I11[0].size, I12[0].size, I13[0].size, I14[0].size, I21[0].size, I22[0].size, I23[0].size, I24[0].size, I31[0].size, I32[0].size, I33[0].size, I34[0].size, I41[0].size, I42[0].size, I43[0].size, I44[0].size, file = OF)

    OF.close()

if __name__ == '__main__':
    evaluate_lenet5()

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
