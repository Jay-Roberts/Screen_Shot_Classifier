from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import os
import multiprocessing as mp


# Define the model function
def my_model_fn(features, labels, mode, config):
    # Input layer
    input_layer = tf.reshape(features, [-1,28,28,3],name='input_layer')
  
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
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
            }
        return tf.estimator.EstimatorSpec( mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Build the function in an EstimatorSpec
def construct_model(config = None):
    return tf.estimator.Estimator(my_model_fn,config=config)



# Helper function to get game tfrecords
def get_name(tag):
    #print(tag)
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
    image = tf.reshape(image, [28, 28,3])

    label = tf.cast(features['label'], tf.int32)
    return image, label


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
    #print(game_IDs)
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
    
    features, labels = iterator.get_next()

    return features, labels


#features = {'image':tf.placeholder(tf.float32,shape = (28,28,3))}
def tmp_server():
    #print('tmp_server here just checking in to say hi...')
    feats = {'image':tf.placeholder(tf.float32,shape = (28,28,3)),
            'label': tf.placeholder(tf.int64)}
    server_fn =  tf.estimator.export.build_raw_serving_input_receiver_fn(feats)
    
    return server_fn

def numpy_input_receiver_fn():
  """Build the serving inputs."""
  features = {'image':tf.placeholder(tf.float32,shape = (28,28,3))}
  return tf.estimator.export.build_raw_serving_input_receiver_fn(features)


# Reads the tf.Example files
def my_mini_parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
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
    image = tf.reshape(image, [28, 28,3])

    return image

def example_serving_input_fn():  

    example_bytestring = tf.placeholder(
        shape=[None],
        dtype=tf.string,
    )
    feature_spec = {'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64)}
    
    #features = tf.parse_example(example_bytestring,feature_spec)

    features = my_parser(example_bytestring)
    reciever_tensors = {'examples':example_bytestring}
    #print('featuressss: ',features)
    return tf.estimator.export.ServingInputReceiver(features,reciever_tensors)


# [START serving-function]
def json_serving_input_fn():
  """Build the serving inputs."""
  inputs1 = tf.placeholder(shape=[28,28,3],dtype=tf.float32)
  inputs2 = {'image': tf.placeholder(shape=[28,28,3],dtype=tf.float32)}

  #return tf.estimator.export.ServingInputReceiver()
  return tf.estimator.export.ServingInputReceiver(inputs2, inputs1)
# [END serving-function]

SERVING_FUNCTIONS = {
    'JSON': json_serving_input_fn,
    'EXAMPLE': example_serving_input_fn,
    #'TMP': tmp_server,
    'TMP': tf.estimator.export.build_raw_serving_input_receiver_fn(
        {'image':tf.placeholder(tf.float32,shape = (28,28,3))}
            )
}

"""
#to test it
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(model_dir='test')
estimator = construct_model(config=run_config)

fd = ['TFRecords']
train_input = lambda: my_input_fn('train',file_dir = fd)
train_spec = tf.estimator.TrainSpec(train_input, max_steps = 100)
exporter = tf.estimator.FinalExporter('testit',tmp_server)

eval_input = lambda: my_input_fn('val',file_dir = fd)
eval_spec = tf.estimator.EvalSpec(eval_input, steps = 100, name = 'eval-it')


tf.logging.set_verbosity('INFO')
tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)"""