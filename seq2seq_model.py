import numpy as np
import tensorflow as tf
from seq2seq import *

class Seq2SeqModel:

    def __init__(self, vocab_size, buckets, size, num_layers, max_gradient_norm,
                 batch_size, learning_rate, learning_rate_decay_factor, use_lstm=False,
                 num_samples=1024, forward_only=False, beam_search=True, beam_size=10, attention=True):
        """Create the model.
        Args:
        vocab_size: size of the source vocabulary.
        buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
        size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.
        use_lstm: if true, we use LSTM cells instead of GRU cells.
        num_samples: number of samples for sampled softmax.
        forward_only: if set, we do not construct the backward pass in the model.
        """
        self.vocab_size = vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate*learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        #If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        #Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples>0 and num_samples<self.vocab_size:
            w = tf.get_variable("proj_w", [size,self.vocab_size], dtype=tf.float32)
            w_t = tf.transpose(w)
            b = tf.get_variable("proj_b", [self.vocab_size], dtype=tf.float32)
            output_projection = (w,b)

            def sampled_loss(labels, inputs):
                labels = tf.reshape(labels, [-1,1])
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.nn.sampled_loss(
                        weights = w_t,
                        biases = b,
                        labels = labels,
                        inputs = local_inputs,
                        num_sampled = num_samples,
                        num_classes = self.vocab_size)
            softmax_loss_function = sampled_loss

            #Create the internal multi-layer cell for our RNN.
            single_cell = tf.contrib.rnn.GRUCell(size)
            if use_lstm:
                single_cell = tf.contrib.rnn.BasicLSTMCell(size)
            cell = single_cell
            if num_layers>1:
                cell = tf.contrib.rnn.MultiRNNCell([single_cell]*num_layers)

            def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
                if attention:
                    print "Attention Model"
                    return embedding_attention_seq2seq(encoder_inputs, decoder_inputs,
                            cell, num_encoder_symbols=vocab_size, num_decoder_symbols=vocab_size,
                            embedding_size=size, output_projection=output_projection,
                            feed_previous=do_decode, beam_search=beam_search, beam_size=beam_size)
                else:
                    print "Basic Model"
                    return embedding_rnn_seq2seq(encoder_inputs, decoder_inputs, cell,
                            num_encoder_symbols=vocab_size, num_decoder_symbols=vocab_size,
                            embedding_size=size, output_projection=output_projection,
                            feed_previous=do_decode, beam_search=beam_search, beam_size=beam_size)
            #Feed inputs
            self.encoder_inputs = []
            self.decoder_inputs = []
            self.target_weights = []
            for i in xrange(buckets[-1][0]): #Last bucket is the biggest one.
                self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{}".format(i)))
            for i in xrange(buckets[-1][1]+2): #The padded length is 2 more than bucket size, see textdata.py
                self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{}".format(i)))
                self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{}".format(i)))

            #Our targets are decoder inputs shifted by one.
            targets = [self.decoder_inputs[i+1] for i in xrange(len(decoder_inputs)-1)]

            if forward_only:
                if beam_search:
                    self.outputs, self.beam_path, self.beam_symbol = decode_model_with_buckets(
                        self.encoder_inputs, self.decoder_inputs, targets,
                        self.target_weights, buckets, lambda x,y: seq2seq_f(x,y,True)
                        softmax_loss_function=softmax_loss_function)
                else:
                    self.outputs, self.losses = model_with_buckets(
                        self.encoder_inputs, self.decoder_inputs, targets,
                        self.target_weights, buckets, lambda x,y: seq2seq_f(x,y,True),
                        softmax_loss_function=softmax_loss_function)
                # If we use output projection, we need to project outputs for decoding.
                if output_projection is not None:
                    for b in xrange(len(buckets)):
                        self.outputs[b] = [
                            tf.matmul(output, output_projection[0])+output_projection[1]
                            for output in self.outputs[b]
                        ]
            else:
                self.outputs, self.losses = model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets, lambda x,y: seq2seq_f(x,y,True),
                    softmax_loss_function=softmax_loss_function)

            #Gradients and SGD update operation for training the model.
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                opt = tf.train.GradientDescentOptimizer(self.learning_rate) #CAN BE REPLACED by a more advanced optimizer
                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params), global_step=self.global_step))
            self.saver = tf.train.Saver(tf.all_variables())
