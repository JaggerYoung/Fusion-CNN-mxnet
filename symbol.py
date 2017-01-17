import find_mxnet
import mxnet as mx

def classify(num_classes):
    input_data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('label')

    fc1 = mx.symbol.FullyConnected(data=input_data, num_hidden=2048)
    relu1 = mx.symbol.Activation(data=fc1, act_type='relu')
    drop1 = mx.symbol.Dropout(data=relu1, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden=2048)
    relu2 = mx.symbol.Activation(data=fc2, act_type='relu')
    drop2= mx.symbol.Dropout(data=relu2, p=0.5)

    fc3 = mx.symbol.FullyConnected(data=drop2, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, label=label, name='softmax')

    return softmax
