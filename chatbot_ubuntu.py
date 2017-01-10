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
from seq2seq_model import *

#hyperparameters
tf.app.flags.DEFINE_float('learning_rate',0.5,'Learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay_factor',0.99,'Learning rate decays by this much')
tf.app.flags.DEFINE_float('max_gradient_norm',5.0,'Clip gradients to this norm')
tf.app.flags.DEFINE_integer('batch_size',64,'Batch size for training')
tf.app.flags.DEFINE_integer('size',5,'Size of each layer') # the number of unfolded LSTM units in each layer(maybe too larger)..? or the dimension of hidden vector?
tf.app.flags.DEFINE_integer('num_layers',2,'Number of layers')
tf.app.flags.DEFINE_integer('num_epochs',30,'maximum number of epochs to run')
tf.app.flags.DEFINE_integer('vocab_size',50000,'Vocabulary size, words with lower frequency are regarded as unknown')
#file paths
tf.app.flags.DEFINE_string('train_samples_path','ubuntu_train_samples.pkl','Processed training samples')
tf.app.flags.DEFINE_string('valid_samples_path','ubuntu_valid_samples.pkl','Processed validation samples')
tf.app.flags.DEFINE_string('test_samples_path','ubuntu_test_samples.pkl','Processed test samples')
tf.app.flags.DEFINE_string('train_data_path','data/ubuntu/train.csv','Training data')
tf.app.flags.DEFINE_string('valid_data_path','data/ubuntu/valid.csv','Validation data')
tf.app.flags.DEFINE_string('test_data_path','data/ubuntu/test.csv','Test data')
tf.app.flags.DEFINE_string('sorted_list_path','ubuntu_freqlist.pkl','List of words sorted by frequency')
tf.app.flags.DEFINE_string('dialog_path','data/ubuntu/dialogs/','Directory of raw ubuntu dialogues')
tf.app.flags.DEFINE_string('train_dir','train','Training directory')
#options
tf.app.flags.DEFINE_integer('playDataset',20,'Display random samples from data')
tf.app.flags.DEFINE_integer('max_train_data_size',0,'Limit on the size of training data (0: no limit)')
tf.app.flags.DEFINE_integer('steps_per_checkpoint',500,'Number of training steps between checkpoints')
tf.app.flags.DEFINE_boolean('decode',False,'Set to True for interactive decoding')
tf.app.flags.DEFINE_integer("beam_size", 100, "???")
tf.app.flags.DEFINE_boolean("beam_search", False, "Set to True for beam_search.")
tf.app.flags.DEFINE_boolean("attention", False, "???")
tf.app.flags.DEFINE_boolean("self_test", False, "Run a self-test if this is set to True.")


FLAGS = tf.app.flags.FLAGS
_buckets = [(10,20),(20,40),(30,60),(40,80),(50,100)]
bucket_id_map = {10:0, 20:1, 30:2, 40:3, 50:4}

def create_model(session, forward_only, beam_search, beam_size=10, attention=True):
    #Create translation model and initialize or load parameters in session.
    model = Seq2SeqModel(FLAGS.vocab_size, _buckets, FLAGS.size, FLAGS.num_layers,
                         FLAGS.max_gradient_norm, FLAGS.batch_size, FLAGS.learning_rate,
                         FLAGS.learning_rate_decay_factor, forward_only=forward_only,
                         beam_search=beam_search, beam_size=beam_size, attention=attention)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train():

    with tf.Session() as sess:

        #Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        #beam_search is not used for training
        model = create_model(sess, False, False)

        #Read data into buckets
        train_set = TextData(FLAGS.train_samples_path, FLAGS.train_data_path, FLAGS.sorted_list_path, FLAGS.dialog_path, FLAGS.vocab_size, FLAGS.playDataset)
        dev_set = TextData(FLAGS.valid_samples_path, FLAGS.valid_data_path, FLAGS.sorted_list_path, FLAGS.dialog_path, FLAGS.vocab_size, FLAGS.playDataset)

        #Training loop
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        for e in range(FLAGS.num_epochs):

            print ("\n")
            print("----- Epoch {}/{} ; (lr={}) -----".format(e+1, FLAGS.num_epochs, model.learning_rate.eval()))

            batches = train_set.get_batches(FLAGS.batch_size)

            tic = datetime.datetime.now()
            for nextBatch in tqdm(batches, desc="Training"):
                #Get a batch and make a step
                start_time = time.time()

                bucket_id = bucket_id_map[len(nextBatch.encoder_inputs)]

                _, step_loss, _ = model.step(sess, nextBatch, bucket_id, False, False) # for nextBatch:  decoderSeqs and targetSeqs are very similar, see DeepQA/textdata.py/Line 121-122

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
                    checkpoint_path = os.path.join(FLAGS.train_dir, "chat_bot.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                    step_time, loss = 0.0, 0.0

                    # Run evals on development set and print their perplexity.
                    dev_batches = dev_set.get_batches(FLAGS.batch_size)
                    eval_loss_sum = 0
                    for dev_batch in dev_batches:
                        bucket_id = bucket_id_map[len(dev_batch.encoder_inputs)]
                        _, eval_loss, _ = model.step(sess, dev_batch, bucket_id, True, False)
                        eval_loss_sum+=eval_loss
                    eval_ppx = math.exp(eval_loss_sum/len(dev_batches)) if eval_loss_sum/len(dev_batches) < 300 else float('inf')
                    print("eval: perplexity %.2f" % eval_ppx)
                    sys.stdout.flush()

            toc = datetime.datetime.now()
            print("Epoch finished in {}".format(toc-tic))

"""def decode():
    with tf.Session() as sess:
        beam_size = FLAGS.beam_size
        beam_search = FLAGS.beam_search
        attention = FLAGS.attention
        model = create_model(sess, True, beam_search, beam_size, attention)

        if beam_search:
            sys.stdout.write("> ")
            sys.stdout.flush()"""

def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()


"""t = TextData('ubuntu_test_samples.pkl', 'data/ubuntu/test.csv', 'ubuntu_freqlist.pkl', 'data/ubuntu/dialogs/', 50000, 0)
b = t.get_batches(50)
print len(b[0].encoder_inputs)
print len(b[0].decoder_inputs)"""
