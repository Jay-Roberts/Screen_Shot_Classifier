import os
import multiprocessing as mp
import glob
import sys
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
from google.cloud import storage
import pandas as pd

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

    return data

# Write train data into a TFRecord
def make_TFRec(gameID):
    
    # Get the data addresses
    data = feed_makeTFR(gameID,source_dir)

    # Can change the split here
    split = (.6,.2,.2) 

    # Upload chunk size
    chunk_size = 500

    # Number of upload tries
    num_tries = 10

    train_size, test_size, val_size = split
    addrs, labels = data
    num_imgs = len(addrs)
    
    names = ['train','test','val']
    
    # Shuffle data for good measure
    c = zip(addrs,labels)
    shuffle(c)
    addrs,labels = zip(*c)
    
    # Create the local temporary path
    local_path = 'TFRecords/%s_tmp'%gameID
    if not os.path.isdir(local_path):
        os.makedirs(local_path)

    # Create the train, test, and val TFRecords
    for name in names:
        name_path = local_path+'/%s_tmp'%name
        # Make the train folders
        if not os.path.isdir(name_path):
            os.makedirs(name_path)
        
        # Pick out the parts for train, test, and val
        if name == 'train':
            ixs = list(range(int(train_size*num_imgs)))

        if name == 'test':
            ixs = list(range(int(train_size*num_imgs),int(train_size*num_imgs)+int(test_size*num_imgs)))
        
        if name == 'val':
            ixs = list(range(int(train_size*num_imgs)+int(test_size*num_imgs),num_imgs))
            #print('val length: %d'%len(ixs))
        
        # Get number of indicies
        num_ixs= len(ixs)
        
        # Chunk it up
        chunks = num_ixs/chunk_size
        orphans = num_ixs%chunk_size

        ixs_blocks = [0]*chunks
        for i in range(chunks):
            ixs_blocks[i] = ixs[i*chunk_size:(i+1)*chunk_size]
        
        if orphans != 0:
            ixs_blocks.append(ixs[-orphans:])
        
        for block in ixs_blocks:
            block_name = ixs_blocks.index(block)+1
            block_name = str(block_name)
            block_size = len(block)

            #open the TFRecords file
            local_filename = name_path+'/%s_%s_tmp.tfrecords'%(name,block_name)

            #Process data and write to the TFRecord
            print('%s: Processing block %s'%(gameID,block_name))
            for i in block:
                # Make a TFRecord Writer object
                writer = tf.python_io.TFRecordWriter(local_filename)

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
        
            # Name the blob path in the bucket
            blob_path = 'TFRecords/%s/%s'%(gameID,name)
        
            blob_filename = blob_path+'/%s_%s_%s.tfrecords'%(gameID,name,block_name)
            local_filename = name_path+'/%s_%s_tmp.tfrecords'%(name,block_name)

            # Make a blobs to upload
            blob = bucket.blob(blob_filename)

            # Upload the TFRecord dir
            # connection issues can get in the way so try a few times
            success = False
            tries = 0
            while (not success and tries < num_tries):
                try:
                    blob.upload_from_filename(local_filename)
                    success = True
                    print('%s: FINISHED Uploading %s-data in %d tries'%(gameID,name,tries+1))
                except:
                    success = False
                    tries+=1
                
            # Remove the local file
            if not success:
                print('%s: Failed to upload %s'%(gameID,local_filename))
            
            os.remove(local_filename)

    # Remove the tmp dirs
    os.removedirs(local_path+'/test_tmp')
    os.removedirs(local_path+'/train_tmp')
    os.removedirs(local_path+'/val_tmp')

    

if __name__ == '__main__':
    # Conncet to google bucket
    client = storage.Client(project='gclass-183202')
    bucket = client.get_bucket('gamepics')

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

    # Write the dictionary to a csv
    labels_col = map(lambda x: game_IDs.index(x), game_IDs)
    labels_df = pd.DataFrame({'GAMEID':game_IDs, 'LABEL': labels_col})

    print('Saving labels')
    labels_df.to_csv('TFRecords/labels_key.csv', index = False)

    # Find how many resources are available
    num_slaves = mp.cpu_count()
    pool = mp.Pool(processes = num_slaves)

    # Put them to work
    pool.map(make_TFRec,game_IDs)