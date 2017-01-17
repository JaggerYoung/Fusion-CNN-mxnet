import os,sys
import random
import find_mxnet
import mxnet as mx
import string
import math

import numpy as np
import cPickle as p
from symbol import classify

BATCH_SIZE = 15

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

    @property
    def provide_data(self):
        return [(n,x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n,x.shape) for n,x in zip(self.label_names, self.label)]


class LRCNIter(mx.io.DataIter):
    def __init__(self, dataset, labelset, num, batch_size):
        
	self.batch_size = batch_size
	self.count = num/batch_size
	self.dataset = dataset
	self.labelset = labelset
	
	self.provide_data = [('data',(batch_size, 1008))]
	self.provide_label = [('label',(batch_size,))]

    def __iter__(self):
	for k in range(self.count):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        idx = k * batch_size + i
		data.append(self.dataset[idx])
		label.append(self.labelset[idx])

	    data_all = [mx.nd.array(data)]
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch

    def reset(self):
        pass

if __name__ == '__main__':
    batch_size = BATCH_SIZE

    num_epoch = 500
    learning_rate = 0.0025
    momentum = 0.0015
    num_classes = 101
    
    train_data_count = 9537
    test_data_count = 3783

    contexts = [mx.context.gpu(0)]

    f_1 = file('train_data.data')
    x_train = p.load(f_1)


    f_2 = file('test_data.data')
    x_test = p.load(f_2)

    f_3 = file('train_label.data')
    y_train = p.load(f_3)
    f_4 = file('test_label.data')
    y_test = p.load(f_4)
     
    #print mx.nd.array(x_train).shape, mx.nd.array(x_test).shape
    #print mx.nd.array(x_test).shape, mx.nd.array(y_test).shape

    data_train = LRCNIter(x_train, y_train, train_data_count, batch_size)
    data_test = LRCNIter(x_test, y_test, test_data_count, batch_size)
    #print data_train.provide_data, data_train.provide_label

    symbol = classify(num_classes)

    model = mx.model.FeedForward(ctx=contexts,
                                 symbol=symbol,
				 num_epoch=num_epoch,
				 learning_rate=learning_rate,
				 momentum=momentum,
				 wd=0.00001,
				 initializer=mx.init.Xavier(factor_type="in",magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    print 'begin fit'
    batch_end_callbacks = [mx.callback.Speedometer(BATCH_SIZE, 1000)]
    eval_metrics = ['accuracy']

    model.fit(X=data_train, eval_data=data_test, eval_metric=eval_metrics, batch_end_callback=batch_end_callbacks)
