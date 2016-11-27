# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cup_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
import tensorflow as tf

import cup_net
from IPython import embed

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/home/chenhaonan/dump/Image/Cup/AlexNet',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('validation_size', 100,
                            """Size of validation data.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            images,
            labels):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  number_of_eval = labels.shape[0]
  steps_per_epoch = number_of_eval // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in range(steps_per_epoch):
    if number_of_eval == FLAGS.batch_size:
      feed_dict = {images_placeholder: images, labels_placeholder: labels}
    else:
      offset = (step * FLAGS.batch_size) % (number_of_eval - FLAGS.batch_size)
      batch_data = images[offset:(offset + FLAGS.batch_size), ...]
      batch_labels = labels[offset:(offset + FLAGS.batch_size)]
      feed_dict = {images_placeholder: batch_data, labels_placeholder: batch_labels}
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))



def train():
  """Train CIFAR-10 for a number of steps."""
  train_data, train_labels = cup_net.load_dataset("train")
  test_data, test_labels = cup_net.load_dataset("valid")

  # Generate a validation set.
  validation_data = train_data[:FLAGS.validation_size, ...]
  validation_labels = train_labels[:FLAGS.validation_size]
  train_data = train_data[FLAGS.validation_size:, ...]
  train_labels = train_labels[FLAGS.validation_size:]
  train_size = train_labels.shape[0]
  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)

    # Get images and labels for CIFAR-10.
    # images, labels = cup_net.inputs(False)
    images, labels = cup_net.distorted_inputs()

    # Build a Graph that computes the logits predictions from the
    # inference model.
    # logits = cup_net.inference(images)
    logits, _ = cup_net.inference_alexnet(images)

    # Calculate loss.
    loss = cup_net.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cup_net.train(loss, global_step)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = cup_net.evaluation(logits, labels)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    # summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    # tf.train.start_queue_runners(sess=sess)

    # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
    # embed()

    for step in range(FLAGS.max_steps):
      start_time = time.time()
      offset = (step * FLAGS.batch_size) % (train_size - FLAGS.batch_size)
      batch_data = train_data[offset:(offset + FLAGS.batch_size), ...]
      batch_labels = train_labels[offset:(offset + FLAGS.batch_size)]
      # This dictionary maps the batch data (as a np array) to the
      # node in the graph it should be fed to.
      feed_dict = {images: batch_data,
                   labels: batch_labels}
      # embed()
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 50 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      if step % 500 == 0:
        do_eval(sess, eval_correct, images, labels, validation_data, validation_labels)
        # summary_str = sess.run(summary_op)
        # summary_writer.add_summary(summary_str, step)
        print("Summary done")

      # Save the model checkpoint periodically.
      if step % 5000 == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)
        do_eval(sess, eval_correct, images, labels, test_data, test_labels)
        print("Checkpoint saved")

    # do_eval(sess, eval_correct, images, labels, test_data, test_labels)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
