'''
Model to predict next sentence given an input sequence
Reference: 
    http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
    
'''
import tensorflow as tf
from text_data_copy import TextData

class Model:    
    def __init__(self, args):
        """
        Args:
            args: parameters of the model
        """
        print("model creation...")
        self.args = args
        self.dtype = tf.float32
        self.args.hiddenSize = 50 # the length of hidden state vector
        self.args.numLayers = 1 # number of layers
    def buildNetwork(self):
        '''
        build the model
        TODO: Use sampled softmax and use projection
        '''
        encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True)
        encoDecoCell = tf.nn.rnn_cell.MultiRNNCell([encoDecoCell] * self.args.numLayers, state_is_tuple=True)


