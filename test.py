import dnn_functions as dnn
import numpy as np

network_obj = dnn.Network([dnn.FullyConnected(2,5),dnn.FullyConnected(5,3)],3)
data = np.asarray(np.random.normal(loc=0.0,scale=1.0,size=(2,1)))
network_obj.full_fwd_prop(data)
label = np.asarray(np.random.normal(loc=0.0,scale=1.0,size=(3,1)))
#print(network_obj.layers[-1].weights)
network_obj.full_back_prop(label)
network_obj.full_fwd_prop(data)
#print(label)
#print(network_obj.layers[-1].weights)