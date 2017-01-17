import find_mxnet
import mxnet as mx
import logging
import time
import cv2
import random
import glob
import numpy as np
import cPickle as p

NUM_SAMPLES = 3
BATCH_SIZE = 1

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
	self.label = label
	self.data_names = data_names
	self.label_names = label_names

	self.pad = 0
	self.index = None

    @property
    def provide_data(self):
        return [(n, x.shape) for n,x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n,x in zip(self.label_names, self.label)]

def readData(Filename, num):
    data_1 = []
    data_2 = []
    f = open(Filename,'r')
    total = f.readlines()
    #print len(total)
    for eachLine in range(len(total)):
        pic = []
        tmp = total[eachLine].split('\n')
	tmp_1, tmp_2 = tmp[0].split(' ',1)
	tmp_1 = '/data/zhigang.yang/UCF-101'+tmp_1
	for filename in glob.glob(tmp_1+'/*.jpg'):
	    pic.append(filename)
	len_pic = len(pic)
	l_n = len_pic/num
	for i in range(num):
	    data_1.append(pic[i*l_n])    
	    data_2.append(int(tmp_2))
    f.close()
    return (data_1, data_2)

def readImg(Filename, data_shape):
    mat = []

    img = cv2.imread(Filename, cv2.IMREAD_COLOR)
    r,g,b = cv2.split(img)
    r = cv2.resize(r, (data_shape[2], data_shape[1]))
    g = cv2.resize(g, (data_shape[2], data_shape[1]))
    b = cv2.resize(b, (data_shape[2], data_shape[1]))
    r = np.multiply(r, 1/255.0)
    g = np.multiply(g, 1/255.0)
    b = np.multiply(b, 1/255.0)

    mat.append(r)
    mat.append(g)
    mat.append(b)

    return mat

class InceptionIter(mx.io.DataIter):
    def __init__(self, fname, num, batch_size, data_shape):
        self.batch_size = batch_size
	self.fname = fname
	self.data_shape = data_shape
	self.num = num*NUM_SAMPLES/batch_size
	(self.data_1, self.data_2) = readData(fname, NUM_SAMPLES)
    
        self.provide_data = [('data', (batch_size,) + data_shape)]
	self.provide_label = [('label', (batch_size,))]

    def __iter__(self):
        for k in range(self.num):
	    data = []
	    label = []
	    for i in range(self.batch_size):
	        idx = k * self.batch_size + i
		img = readImg(self.data_1[idx], self.data_shape)
		data.append(img)
	        label.append(self.data_2[idx])
	
	    data_all = [mx.nd.array(data)]
	    label_all = [mx.nd.array(label)]
	    data_names = ['data']
	    label_names = ['label']

	    data_batch = SimpleBatch(data_names, data_all, label_names, label_all)
	    yield data_batch
    
    def reset(self):
        pass

if __name__ == '__main__':
#def vgg_predict():    
    train_num = 9537
    test_num = 3783

    batch_size = BATCH_SIZE
    data_shape = (3, 299, 299)
    
    train_file = '/home/users/zhigang.yang/mxnet/example/LRCN-for-Activity-Recognition/data/train.list'
    test_file = '/home/users/zhigang.yang/mxnet/example/LRCN-for-Activity-Recognition/data/test.list'

    
    data_train = InceptionIter(train_file, train_num, batch_size, data_shape)
    data_val = InceptionIter(test_file, test_num, batch_size, data_shape)

    print data_train.provide_data, data_train.provide_label

    devs = [mx.context.gpu(2)]
    model = mx.model.FeedForward.load("./googlenet_model/Inception-7", epoch=0001, ctx=devs, num_batch_size=BATCH_SIZE)

    internals = model.symbol.get_internals()
    print internals.list_outputs()
    fea_symbol = internals['fc1_output']
    feature_exactor = mx.model.FeedForward(ctx=devs, symbol=fea_symbol, num_batch_size=1,
                                           arg_params=model.arg_params, aux_params=model.aux_params,
    					   allow_extra_params=True)
    train_result = feature_exactor.predict(data_train)
    test_result = feature_exactor.predict(data_val)
    vgg_train_result = []
    vgg_test_result = []
    for i in range(len(train_result)/3):
	vgg_train_result.append([train_result[3*i]+train_result[3*i+1]+train_result[3*i+2] for i in range(len(train_result[0]))])
	vgg_test_result.append([test_result[3*i]+test_result[3*i+1]+test_result[3*i+2] for i in range(len(test_result[0]))])

    print mx.nd.array(vgg_train_result).shape
    print mx.nd.array(vgg_test_result).shape
    #return (vgg_train_result, vgg_test_result)
    train_data_file = 'train_data.data'
    f_1 = file(train_data_file, 'w')
    p.dump(vgg_train_result, f_1)
    f_1.close()

    test_data_file = 'test_data.data'
    f_2 = file(test_data_file, 'w')
    p.dump(vgg_test_result, f_2)
    f_2.close()

#def get_label():
    
    (tmp_1, train_label_1) = readData(train_file, NUM_SAMPLES)
    (tmp_2, test_label_1) = readData(test_file, NUM_SAMPLES)
    train_label = []
    test_label = []
    for i in range(len(train_label_1)/3):
        train_label.append(train_label_1[i*3])
        test_label.append(test_label_1[i*3])

    print mx.nd.array(train_label).shape
    print mx.nd.array(test_label).shape
#   return (train_label, test_label)
    train_label_file = 'train_label.data'
    f_3 = file(train_label_file, 'w')
    p.dump(train_label, f_3)
    f_3.close()

    test_label_file = 'test_label.data'
    f_4 = file(test_label_file, 'w')
    p.dump(test_label, f_4)
    f_4.close()

