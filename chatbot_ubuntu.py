import math
import os
import random
import sys
import time
import numpy as np
import tensorflow as tf
import datetime
from tqdm import tqdm

from text_data import *
from create_dict import *
from model_seq2seq import Model

tf.app.flags.DEFINE_float('learning_rate',0.5,'Learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor',0.99,'Learning rate decays by this much')
tf.app.flags.DEFINE_float('max_gradient_norm',5.0,'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size',64,'Batch size for training')
tf.app.flags.DEFINE_integer('size',512,'Size of each layer') # the number of unfolded LSTM units in each layer(maybe too larger)..? or the dimension of hidden vector?
tf.app.flags.DEFINE_integer('num_layers',3,'Number of layers')
tf.app.flags.DEFINE_integer('num_epochs',30,'maximum number of epochs to run')
tf.app.flags.DEFINE_integer('vocab_size',50000,'Vocabulary size, words with lower frequency are regarded as unknown')
tf.app.flags.DEFINE_string('train_samples_path','ubuntu_train_samples.pkl','Processed training samples')
tf.app.flags.DEFINE_string('valid_samples_path','ubuntu_valid_samples.pkl','Processed validation samples')
tf.app.flags.DEFINE_string('test_samples_path','ubuntu_test_samples.pkl','Processed test samples')
tf.app.flags.DEFINE_string('train_data_path','data/ubuntu/train.csv','Training data')
tf.app.flags.DEFINE_string('valid_data_path','data/ubuntu/valid.csv','Validation data')
tf.app.flags.DEFINE_string('test_data_path','data/ubuntu/test.csv','Test data')
tf.app.flags.DEFINE_string('sorted_list_path','ubuntu_freqlist.pkl','List of words sorted by frequency')
tf.app.flags.DEFINE_string('dialog_path','data/ubuntu/dialogs/','Directory of raw ubuntu dialogues')
tf.app.flags.DEFINE_integer('playDataset',20,'Display random samples from data')
tf.app.flags.DEFINE_integer('max_train_data_size',0,'Limit on the size of training data (0: no limit)')
tf.app.flags.DEFINE_integer('steps_per_checkpoint',500,'Number of training steps between checkpoints')
tf.app.flags.DEFINE_boolean('decode',False,'Set to True for interactive decoding')

FLAGS = tf.app.flags.FLAGS
_buckets = [(10,20),(20,40),(30,60),(40,80),(50,100)]

def create_model(session):
    model = Model(FLAGS)
    # TODO: save and restore model to/from hard disk
    Load_model = False
    if Load_model:
        model.saver.restore(session, FILA_NAME)
    else:
        session.run(tf.initialize_all_variables())
    return model
    
    

def train():

    with tf.Session() as sess:

        #Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess)

        #Read data into buckets
        train_set = TextData(FLAGS.train_samples_path, FLAGS.train_data_path, FLAGS.sorted_list_path, FLAGS.dialog_path, FLAGS.vocab_size, FLAGS.playDataset)
        dev_set = TextData(FLAGS.valid_samples_path, FLAGS.valid_data_path, FLAGS.sorted_list_path, FLAGS.dialog_path, FLAGS.vocab_size, FLAGS.playDataset)

        #Training loop
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_loss = []

        for e in range(FLAGS.num_epochs):

            print ("\n")
            print("----- Epoch {}/{} ; (lr={}) -----".format(e+1, FLAGS.num_epochs, model.learning_rate.eval()))

            batches = train_set.get_batches(FLAGS.batch_size)

            tic = datetime.datetime.now()
            for nextBatch in tqdm(batches, desc="Training"):
                #Get a batch and make a step
                start_time = time.time()
                step_loss, ??? = model.step(sess, nextBatch.encoder_inputs, nextBatch.decoder_inputs, nextBatch.weights, ???)
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += step_loss / FLAGS.steps_per_checkpoint
                current_step += 1

                # Save checkpoints, print statistics and run evals
                if current_step%FLAGS.steps_per_checkpoint==0:
                    #Print statistics for the previous epoch.
                    perplexity = math.exp(loss) if loss < 300 else float('inf')
                    print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                    #Decrease learning rate if no improvement was seen over last 3 times.
                    if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                        sess.run(model.learning_rate_decay_op)
                    previous_losses.append(loss)
                    # Save checkpoint and zero timer and loss.
                    model.saver.save(sess, "chat_bot.ckpt", global_step=model.global_step)
                    step_time, loss = 0.0, 0.0

                    # Run evals on development set and print their perplexity.
                    dev_batches = dev_set.get_batches(FLAGS.batch_size)
                    eval_loss_sum = 0
                    for dev_batch in dev_batches:
                        eval_loss, ??? = model.step(sess, dev_batch.encoder_inputs, dev_batch.decoder_inputs, dev_batch.weights, ???)
                        eval_loss_sum+=eval_loss
                    eval_ppx = math.exp(eval_loss_sum/len(dev_batches)) if eval_loss_sum/len(dev_batches) < 300 else float('inf')
                    print("  eval: perplexity %.2f" % eval_ppx)
                    sys.stdout.flush()

            toc = datetime.datetime.now()
            print("Epoch finished in {}".format(toc-tic))
