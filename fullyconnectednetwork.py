import dnn_functions as dnn
import numpy as np
import _pickle as cPickle
import gzip

f = gzip.open('data\mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
f.close()

network_obj = dnn.Network([dnn.FullyConnected(784,15),dnn.FullyConnected(15,10)],3)
for k in range(1000):
    for i in range(len(training_data[1])):
        training_data_point = training_data[0][i].reshape([784,1])
        network_obj.full_fwd_prop(training_data_point)
        label = np.zeros((10,1)); 
        label[training_data[1][i]] = 1.0
        network_obj.full_back_prop(label)
    print(k)

correct = 0
for i in range(len(validation_data[1])):
   validation_data_point = validation_data[0][i].reshape([784,1])
   network_obj.full_fwd_prop(validation_data_point)
   if i == 998:
    print(np.argmax(network_obj.layers[-1].a))
    print(validation_data[1][i])
   if np.argmax(network_obj.layers[-1].a) == validation_data[1][i]:
       correct += 1

print(f'Accuracy: {correct/len(validation_data[1])}')
