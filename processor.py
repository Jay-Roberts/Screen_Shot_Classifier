import os
import multiprocessing
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
    img = img.astype(np.float32)
    return img

# Make Feature functions for TFRecord format
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [value]))

# Write train data into a TFRecord
def make_TFRec(data, path, split = (.6,.2,.2) ):
    """
    data = [X,Y] X address of image files, Y labels for the images
    path = the save path
    """
    train_size, test_size, val_size = split
    addrs, labels = data
    num_imgs = len(addrs)

    names = ['train','test','val']
    
    # Shuffle data for good measure
    c = zip(addrs,labels)
    shuffle(c)
    addrs,labels = zip(*c)
    

    for name in names:
        #open the TFRecords file
        filename = path+'/'+name+'.tfrecords' #address to save TFRecords file
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
    
    # Get the game list
    game_IDs = os.listdir(source_dir)

    # Make keys
    labels_key = {x: game_IDs.index(x) for x in game_IDs}

    # Get all addresses and labels
    jobs = []
    for game in labels_key.keys():
        
        # Make the game path
        path = source_dir+'/%s'%game
        gameID = labels_key[game]

        # Get the game's img addresses and its label
        addrs = glob.glob(path+'/*.jpg')
        num_imgs = len(addrs)
        labels = [gameID]*num_imgs
        
        # Run the TFRecod function on the data
        make_TFRec([addrs,labels],path)
        p = multiprocessing.Process(target=make_TFRec, args=([addrs,labels],path))
        jobs.append(p)
        p.start()

    
    
        