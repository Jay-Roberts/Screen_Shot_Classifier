from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
import multiprocessing as mp


# Define the model function
def my_model_fn(features,mode, config):
    print('MODE:',mode)
    #print('Feature type: ', features.shape)
    # Input layer
    input_layer = tf.reshape(features['image'], [-1,28,28,3],name='input_layer')
    labels = features['label']
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    # Units is number of games
    logits = tf.layers.dense(inputs=dropout, units=5)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1, name = "class"),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    
    # Calculate Loss (for both TRAIN and EVAL modes)
    # Depth is number of games
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, 
                                            predictions=predictions["classes"])
            }
        return tf.estimator.EstimatorSpec( mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        # Make a PredictOutput object.
        # predcitions dictionary keys = {'classes','predicitions'}
        predict_out = tf.estimator.export.PredictOutput(predictions)
        exp_ops = {'pres':predict_out}
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions,export_outputs=exp_ops)


# Helper function to get game tfrecords
def get_name(tag):
    
    game,name,file_dir = tag
    result = []
    path = file_dir+'/%s/%s'%(game,name)
    game_folders = os.listdir(path)
    
    for x in game_folders:
        result.append(path+'/'+x)
    
    return result

# joins lists of lists
def join_list(l):
    result = []
    for x in l:
        result+=x
    return result

# Reads the tf.Example files
def my_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    print('HERE', serialized_example)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.decode_raw(features['image'], tf.float32)

    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    image = tf.cast(image, tf.float32) / 255 - 0.5

    # Real small now
    image =  tf.reshape(image, [28, 28,3])

    label = tf.cast(features['label'], tf.int32)
    return {'image':image,'label': label}


# The input function
def my_input_fn(name,file_dir = ['TFRecords'],
                    num_epochs = None,
                    shuffle = True,
                    batch_size = 100):
    # Get games
    file_dir = file_dir[0]
    
    game_IDs = os.listdir(file_dir)
    num_games = len(game_IDs)
    files = [file_dir]*num_games
    
    # Get tfrecords
    if name == 'train':
        games = zip(game_IDs,['train']*num_games,files)
    if name == 'test':
        games = zip(game_IDs,['test']*num_games,files)
    if name == 'val':
        games = zip(game_IDs,['val']*num_games,files)

    filenames = list(map(get_name,games))
    filenames = join_list(filenames)
    

    # Import image data
    dataset = tf.data.TFRecordDataset(filenames)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size = batch_size)

    # Map the parser over dataset, and batch results by up to batch_size
    # Shuffling and batching can be slow
    # get more resources
    num_slaves = mp.cpu_count()
    dataset = dataset.map(my_parser,num_parallel_calls=num_slaves)
    
    # Buffer the batch size of data
    dataset = dataset.prefetch(batch_size)

    # Batch it and make iterator
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    
    features = iterator.get_next()

    return features


