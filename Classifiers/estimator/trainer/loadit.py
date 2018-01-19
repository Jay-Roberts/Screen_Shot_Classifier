import tensorflow as tf
import numpy as np
import argparse
import os



# Used for scanning for latest checkpoint, which is passed as a command line argument
parser = argparse.ArgumentParser(description='Load the latest Saved trained Nueral Netowrk')
parser.add_argument('-d', '--ckpt_file', metavar='REL_PATH', type=str,
                   help='the relative path for the checkpoint file')
args = parser.parse_args()


# Load the metagraph and variables
graph_path = os.getcwd() + args.ckpt_file
print(graph_path)
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    # Load the graph with the trained states
    loader = tf.train.import_meta_graph(graph_path + '.meta')
    loader.restore(sess, graph_path)

    print('Neural Network Successfully Loaded')

    input = graph.get_operation_by_name("input_layer").outputs[0]
    prediction=graph.get_operation_by_name("softmax_tensor").outputs[0]
    classes = graph.get_tensor_by_name("class:0")
    
    print(input,prediction)
    newdata= np.zeros((28,28,3))
    sess.run([prediction,classes],feed_dict={input:[newdata]})
    
