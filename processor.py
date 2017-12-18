import os
import multiprocessing as mp
import glob
import sys
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
  

# Loads images
def load_image(addr):
    """addr: the filepath of an image file
    returns a np.float32 array representation of the image. Here is where you can add image 
    processing."""
    # cv2 loads images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = img.astype(np.float32)
    return img

# Make Feature functions for TFRecord format
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [value]))


# Take gameID and get input to feed the TFRecod maker
def feed_makeTFR(gameID, source_dir):
    # Make the game path
    path = source_dir+'/%s'%gameID
    game_label = labels_dict[gameID]

    # Get the game's img addresses and its label
    addrs = glob.glob(path+'/*.jpg')
    num_imgs = len(addrs)
    labels = [game_label]*num_imgs
    
    data = [addrs,labels]

    return [data,path]

# Write train data into a TFRecord
def make_TFRec(gameID):

    data,path = feed_makeTFR(gameID,source_dir)
    # Can change the split here
    split = (.6,.2,.2) 

    train_size, test_size, val_size = split
    addrs, labels = data
    num_imgs = len(addrs)
    
    names = ['train','test','val']
    
    # Shuffle data for good measure
    c = zip(addrs,labels)
    shuffle(c)
    addrs,labels = zip(*c)
    
    if not os.path.isdir('TFRecords/'+gameID):
        os.makedirs('TFRecords/'+gameID)
    for name in names:
        #open the TFRecords file
        filename = 'TFRecords/%s/%s.tfrecords'%(gameID,name) #address to save TFRecords file
        writer = tf.python_io.TFRecordWriter(filename)

        # Pick out the parts for train, test, and val
        if name == 'train':
            ixs = list(range(int(train_size*num_imgs)))

        if name == 'test':
            ixs = list(range(int(train_size*num_imgs),int(train_size*num_imgs)+int(test_size*num_imgs)))
        
        if name == 'val':
            ixs = list(range(int(train_size*num_imgs)+int(test_size*num_imgs),num_imgs))

        num_data = len(ixs)
        #Process data
        for i in ixs:
            # Progress check. Low for testing
            if not i % 150:
                print('%s-%s data: %d/%d'%(path,name,i,num_data))
                sys.stdout.flush
        
            # Load the image
            img = load_image(addrs[i])
            label = labels[i]

            # Create feature
            feature = {
                name+'/label':_int64_feature(label),
                name+'/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
                }
            
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()
        sys.stdout.flush()

if __name__ == '__main__':
    # Create a list of file addresses and their labels
    # Where to pull from
    source_dir = 'GameImages'

    # Where to put them
    if not os.path.isdir('TFRecords'):
        os.makedirs('TFRecords')
    
    # Get the game list
    game_IDs = os.listdir(source_dir)

    # Make keys
    labels_dict = {x: game_IDs.index(x) for x in game_IDs}

    # Find how many resources are available
    num_slaves = mp.cpu_count()
    pool = mp.Pool(processes = num_slaves)
    pool.map(make_TFRec,game_IDs)

    
        