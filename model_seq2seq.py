'''
Model to predict next sentence given an input sequence
Reference: 
    http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
    
'''
import tensorflow as tf
from text_data_copy import TextData

class Model:    
    def __init__(self, FLAGS):
        """
        Args:
            args: parameters of the model
        """
        print("model creation...")
        
        self.args.numLayers = FLAGS.num_layers # number of layers
        self.args.size = FLAGS.size # the number of unfolded LSTM units in each layer
        self.args.vocab_size = FLAGS.vocab_size
        self.args.test = FLAGS.testMode # true if test (we use previous output as next input (feed_previous) )
        self.args.learning_rate = FLAGS.learning_rate
        
        self.args.hiddenSize = 50 # the length of hidden state vector
        self.args.softmaxSamples = 1024 # the number of samples for sampled softmax. (seq2seq_model)
        self.encoderInputs = None
        self.decoderInputs = None
        self.decoderTargets = None
        self.decoderWeights = None
        self.args.embeddingSize = 32 # the size of word embedding, used in tf.nn.seq2seq.embedding_rnn_seq2seq (DeepQA) [in seq2seq it is set same with hidden vector size]
        
        
        self.buildNetwork()
    def buildNetwork(self):
        '''
        build the model
        TODO: Use sampled softmax and use projection
        '''
        encoDecoCell = tf.nn.rnn_cell.BasicLSTMCell(self.args.hiddenSize, state_is_tuple=True)
        # TODO: Add dropout: tf.nn.rnn_cell.DropoutWrapper()
        encoDecoCell = tf.nn.rnn_cell.MultiRNNCell([encoDecoCell] * self.args.numLayers, state_is_tuple=True)
        
        with tf.name_scope('placeholder_encoder'):
            self.encoderInputs  = [tf.placeholder(tf.int32,   [None, ]) for _ in range(self.args.maxLengthEnco)]  # Batch size * sequence length * input dim
        with tf.name_scope('placeholder_decoder'):
            self.decoderInputs  = [tf.placeholder(tf.int32,   [None, ], name='inputs') for _ in range(self.args.maxLengthDeco)]  # Same sentence length for input and output (Right ?)
            self.decoderTargets = [tf.placeholder(tf.int32,   [None, ], name='targets') for _ in range(self.args.maxLengthDeco)]
            self.decoderWeights = [tf.placeholder(tf.float32, [None, ], name='weights') for _ in range(self.args.maxLengthDeco)]
        
        decoderOutputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(
            self.encoderInputs,
            self.decoderInputs,
            encoDecoCell,
            self.args.vocab_size,
            self.args.vocab_size,
            embedding_size=self.args.embeddingSize,
            output_projection= None, # TODO: use projection to speed up training?
            feed_previous=bool(self.args.test)
            )
        if self.args.test:
            # TODO: Add projection?
            self.outputs = decoderOutputs
        else:
            self.lossFct = tf.nn.seq2seq.sequence_loss(
                decoderOutputs,
                self.decoderTargets,
                self.decoderWeights,
                self.args.vocab_size,
                softmax_loss_function = None # the default softmax (should projection/sample to speed up training)
            )
            tf.scalar_summary('loss', self.lossFct)
            opt = tf.train.AdamOptimizer(
                learning_rate=self.args.learningRate,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-08
            )
            self.optOp = opt.minimize(self.lossFct)
    def step(self):
        
