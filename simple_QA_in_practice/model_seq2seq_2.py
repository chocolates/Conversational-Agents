'''
Model to predict next sentence given an input sequence
Reference: 
    http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
    
'''
import tensorflow as tf
from text_data_2 import TextData

class Model:    
    def __init__(self, FLAGS):
        """
        Args:
            args: parameters of the model
        """
        print("model creation...")
        print FLAGS.num_layers
        self.numLayers = FLAGS.num_layers # number of layers
        self.size = FLAGS.size # the number of unfolded LSTM units in each layer
        self.maxLengthEnco = self.size
        self.maxLengthDeco = self.size
        self.vocab_size = FLAGS.vocab_size
        # self.test = FLAGS.testMode # true if test (we use previous output as next input (feed_previous) )
        self.test = False
        self.learning_rate = FLAGS.learning_rate
        
        self.hiddenSize = 50 # the length of hidden state vector
        self.softmaxSamples = 1024 # the number of samples for sampled softmax. (seq2seq_model)
        self.encoderInputs = None
        self.decoderInputs = None
        self.decoderTargets = None
        self.decoderWeights = None
        self.embeddingSize = 32 # the size of word embedding, used in tf.nn.seq2seq.embedding_rnn_seq2seq (DeepQA) [in seq2seq it is set same with hidden vector size]
        self.learningRate = FLAGS.learning_rate
        print "starting building network..."
        self.buildNetwork()
        print "network built..."
    def buildNetwork(self):
        '''
        build the model
        TODO: Use sampled softmax and use projection
        '''
        encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(self.hiddenSize, state_is_tuple=True)
        # TODO: Add dropout: tf.nn.rnn_cell.DropoutWrapper()
        encoDecoCell = tf.nn.rnn_cell.MultiRNNCell([encoDecoCell] * self.numLayers, state_is_tuple=True)
        
        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.maxLengthEnco)]  # Batch size * sequence length * input dim
        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.maxLengthDeco)]
        
        

        decoderOutputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,
            self.decoderInputs,
            encoDecoCell,
            self.vocab_size,
            self.vocab_size,
            embedding_size=self.embeddingSize, # DeepQA/chatbot.py/loadEmbedding: 
            output_projection= None,
            feed_previous=False)
        if self.test:
            # TODO: Add projection?
            self.outputs = decoderOutputs
        else:
            self.lossFct = tf.nn.seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.vocab_size,
                softmax_loss_function = None # the default softmax (should projection/sample to speed up training)
            )
            tf.scalar_summary('loss', self.lossFct)
            opt = tf.train.AdamOptimizer(
                learning_rate=self.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct) # the operator computing loss function
            
    def step(self, batch):
        feedDict = {}
        ops = None
        
        if not self.test:
            for i in range(self.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            for i in range(self.maxLengthDeco):
                feedDict[self.decoderInputs[i]]  = batch.decoderSeqs[i] # Ref: DeepQA/model.py/step(); DeepQA/textdata.py/line 121 and line 122 
                feedDict[self.decoderTargets[i]] = batch.targetSeqs[i]
                feedDict[self.decoderWeights[i]] = batch.weights[i]
            ops = (self.optOp, self.lossFct) # self.optOp is needed, because we need to minimize self.lossFct
        else:
            for i in range(self.maxLengthEnco):
                feedDict[self.encoderInputs[i]]  = batch.encoderSeqs[i]
            feedDict[self.decoderInputs[0]]  = [self.textData.goToken]
            
            ops = (self.outputs,) # tuple
        return ops, feedDict
