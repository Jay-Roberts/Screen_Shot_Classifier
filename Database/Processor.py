import os
import argparse
import multiprocessing as mp
import glob
import sys
import tensorflow as tf
import numpy as np
from random import shuffle
import cv2
import pandas as pd
import datetime

  
# Loads images
def load_image(addr,res):
    """
    Loads an image from a file as a float32 numpy array. Here is where you can add image 
    processing.
    Inputs:
        addr: The file path to the image file. (str)
        res: Desired resolution. (list)
    Returns:
        Resized image as numpy array shape (res,3) with dtype np.float32
    """
    res = tuple(res)
    # cv2 loads images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img,res)
    img = img.astype(np.float32)
    return img

# Make Feature functions for TFRecord format
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value = [value]))

# Write train data into a TFRecord

#
# TO DO:
# Add in the google bucket compatability
#
def make_TFRec(gameID,source_dir,save_dir,hyp_args,labels_dict):
    """
    Makes a TFRecord from image files. Splits images into train,test, and val files.
    Saves to:
            save_dir/gameID/name/name_tag.tfrecor
    where name is train,test, or val and tag is an int identifier.
    Inputs:
        gameID: The gameID for labeling purposes. (str)
        source_dir: The path to the image files. (str)
        save_dir: The path to save to. (str)
        hyp_args: (split,res,chunk_size). (3-tuple, list, int)
            split: The (train,test,val) split for the data. Must add to 1. (3-tuple floats)
            res: Desired resolution M x N . (list)
            chunk_size: Size to chunk images for upload. (int)
    Returns:
        None.
    """
    #print(labels_dict)

    # Unpack hyper arguments
    split,res,chunk_size = hyp_args
    
    # Make the game path
    source_path = source_dir+'/%s'%gameID
    save_path = save_dir+'/%s'%gameID
    game_label = labels_dict[gameID]

    # Get the game's img addresses and its label
    addrs = glob.glob(source_path+'/*.jpg')
    num_imgs = len(addrs)
    labels = [game_label]*num_imgs

    # split the data
    names = ['train','test','val']
    train_size, val_size, test_size = split
    
    # Shuffle data for good measure
    c = list(zip(addrs,labels))
    shuffle(c)
    addrs,labels = zip(*c)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for name in names:
        name_path = save_path+'/'+name
        
        if not os.path.isdir(name_path):
            os.makedirs(name_path)
        
        tag = len(os.listdir(name_path))
        tag = str(tag)
        #open the TFRecords file
        filename = name_path+'/'+name+'_'+tag+'.tfrecords' #address to save TFRecords file
        #writer = tf.python_io.TFRecordWriter(filename)

        # Pick out the parts for train, test, and val
        if name == 'train':
            ixs = list(range(int(train_size*num_imgs)))

        if name == 'test':
            ixs = list(range(int(train_size*num_imgs),int(train_size*num_imgs)+int(test_size*num_imgs)))
        
        if name == 'val':
            ixs = list(range(int(train_size*num_imgs)+int(test_size*num_imgs),num_imgs))

        num_data = len(ixs)

        # Chunk up the idecies
        num_ixs= len(ixs)
        
        # Chunk it up
        chunks = num_ixs//chunk_size
        orphans = num_ixs%chunk_size

        ixs_blocks = [0]*chunks

        for i in range(chunks):
            ixs_blocks[i] = ixs[i*chunk_size:(i+1)*chunk_size]
        
        if orphans != 0:
            ixs_blocks.append(ixs[-orphans:])
        
        # Find how many records are already in the directory
        num_old_recs = len(os.listdir(save_path+'/'+name))

        for block in ixs_blocks:
            block_name = num_old_recs + ixs_blocks.index(block)
            block_name = str(block_name)
            block_size = len(block)

            local_filename = name_path+'/%s_%s.tfrecords'%(name,block_name)
            writer = tf.python_io.TFRecordWriter(local_filename)
            print('%s: Processing block %s'%(gameID,block_name))
            #Process data
            for i in ixs:
                # Progress check. Low for testing
                if  i % 150 == 0:
                    print('%s-%s data: %d/%d'%(name_path,name,ixs.index(i),num_data))
                    sys.stdout.flush
            
                # Load the image
                img = load_image(addrs[i],res)
                label = labels[i]

                # Create feature
                feature = {
                    'label':_int64_feature(label),
                    'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
                    }
                
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()
            sys.stdout.flush()


# Unpack it
def make_TFRec_unpack(gameID_source_dir_save_dir_split_res_labels_dict):
    gameID,source_dir,save_dir,hyp_args,labels_dict = gameID_source_dir_save_dir_split_res_labels_dict
    print(gameID_source_dir_save_dir_split_res_labels_dict)
    make_TFRec(gameID,source_dir,save_dir,hyp_args,labels_dict)

def make_TFRec_cld(gameID,source_dir,save_dir,hyp_args,knocks):
    """
    Modification of make_TFRec to save processed images to Google Cloud Bucket. 
    Inputs:
        gameID: The gameID for labeling purposes. (str)
        source_dir: The path to the image files. (str)
        save_dir: The path to save to. (str)
        hyp_args: (split,res,chunk_size). (3-tuple, list, int)
            split: The (train,test,val) split for the data. Must add to 1. (3-tuple floats)
            res: Desired resolution M x N . (list)
            chunk_size: Size to chunk images for upload. (int)
        knocks: Number of attempts to upload blob chunks to bucket. (int)
    Returns:
        None.
    """
    # Unpack hyper arguments
    split,res,chunk_size = hyp_args
    
    # Make the game path
    source_path = source_dir+'/%s'%gameID
    save_path = save_dir+'/%s'%gameID
    game_label = labels_dict[gameID]

    # Get the game's img addresses and its label
    addrs = glob.glob(source_path+'/*.jpg')
    num_imgs = len(addrs)
    labels = [game_label]*num_imgs

    # split the data
    names = ['train','test','val']
    train_size, val_size, test_size = split
    
    # Shuffle data for good measure
    c = zip(addrs,labels)
    shuffle(c)
    addrs,labels = zip(*c)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for name in names:
        name_path = save_path+'/'+name
        
        if not os.path.isdir(name_path):
            os.makedirs(name_path)
        #open the TFRecords file
        filename = name_path+'/'+name+'.tfrecords' #address to save TFRecords file

        # Pick out the parts for train, test, and val
        if name == 'train':
            ixs = list(range(int(train_size*num_imgs)))

        if name == 'test':
            ixs = list(range(int(train_size*num_imgs),int(train_size*num_imgs)+int(test_size*num_imgs)))
        
        if name == 'val':
            ixs = list(range(int(train_size*num_imgs)+int(test_size*num_imgs),num_imgs))

        num_data = len(ixs)

        # Chunk up the idecies
        num_ixs= len(ixs)
        
        # Chunk it up
        chunks = num_ixs/chunk_size
        orphans = num_ixs%chunk_size

        ixs_blocks = [0]*chunks

        for i in range(chunks):
            ixs_blocks[i] = ixs[i*chunk_size:(i+1)*chunk_size]
        
        if orphans != 0:
            ixs_blocks.append(ixs[-orphans:])
        
        # Process the blocks
        for block in ixs_blocks:
            # make a unique name
            block_name = datetime.datetime.now().time()
            block_name = list(str(block_name))
            block_name = [e for e in block_name if e not in (' ',':','.','-')]
            block_name = ''.join(block_name)

            block_size = len(block)

            local_filename = name_path+'/%s_%s.tfrecords'%(name,ixs_blocks.index(block)+1)
            writer = tf.python_io.TFRecordWriter(local_filename)
            print('%s: Processing block %s'%(gameID,block_name))
            
            #Process data in TFRecord format
            for i in ixs:
                # Progress check. Low for testing
                if  i % 150 == 0:
                    print('%s-%s data: %d/%d'%(gameID,name,ixs.index(i),num_data))
                    sys.stdout.flush
            
                # Load the image
                img = load_image(addrs[i],res)
                label = labels[i]

                # Create feature
                feature = {
                    'label':_int64_feature(label),
                    'image': _bytes_feature(tf.compat.as_bytes(img.tostring()))
                    }
                
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()
            sys.stdout.flush()
            
            # Name the blob path in the bucket
            blob_path = save_dir+'/%s/%s'%(gameID,name)

            # Make a blobs to upload
            blob = bucket.blob(blob_filename)

            # Upload the TFRecord dir
            # connection issues can get in the way so try a few times
            
            success = False
            knock = 0
            while (not success and knock < knocks):
                try:
                    blob.upload_from_filename(local_filename)
                    success = True
                    print('%s: FINISHED Uploading %s-data in %d tries'%(gameID,name,tries+1))
                except:
                    knock+=1
                
            # Remove the local file
            if not success:
                print('%s: Failed to upload %s'%(gameID,local_filename))
            
            os.remove(local_filename)

# Unpack it
def make_TFRec_cld_unpack(gameID_source_dir_save_dir_hyp_args_knocks):
    gameID,source_dir,save_dir,hyp_args,knocks = gameID_source_dir_save_dir_hyp_args_knocks
    make_TFRec_cld(gameID,source_dir,save_dir,hyp_args,knocks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Local Arguments
    parser.add_argument('--image-dir',
        help='The name of directory with the urls',
        default='GameImages',
        type=str
    )

    parser.add_argument('--save-dir',
        help='The name of file directory to store the url lists',
        default='TFRecords',
        type=str
    )

    parser.add_argument('--num-cores',
        help='The number of cores to use.',
        default=mp.cpu_count(),
        type=int
    )

    # Hyper Arguments
    parser.add_argument('--resolution',
            help='Desired resolution. 2 arguments needed. Default is 28 x 28',
            default=[28,28],
            nargs=2,
            type=int
    )
    
    parser.add_argument('--split',
            help='Percent split between train, eval, and test resepectively. Should add to 1',
            default=[.6,.2,.2],
            nargs=3,
            type=float
    )
    
    parser.add_argument('--chunk-size',
        help='The size of image chunk to make each TFRecord',
        default=100,
        type=int
    )

    # GCLOUD Arguments
    parser.add_argument('--g-project',
        help='The name of your gcloud project',
        type=str
    )

    parser.add_argument('--g-bucket',
            help='The name of an existing gcloud bucket for upload. Required if G_CLOUD is set.',
            type=str
    )

    parser.add_argument('--knocks',
        help='Number of upload attempts for a blob. Default 4',
        default=4,
        type=int
    )

    args = parser.parse_args()

    # Main arguments
    source_dir = args.image_dir
    save_dir = args.save_dir
    num_slaves = args.num_cores
    
    # Hyper arguments
    hyp_arg = [args.split,args.resolution,args.chunk_size]

    # GCLOUD arguments
    g_project = args.g_project
    g_bucket = args.g_bucket
    knocks = args.knocks

    # Find all the games
    game_IDs = os.listdir(source_dir)
    #print(game_IDs)

    # Make keys
    #global labels_dict
    labels_dict = {x: game_IDs.index(x) for x in game_IDs}

    # Write the dictionary to a csv
    labels_col = list(map(lambda x: game_IDs.index(x), game_IDs))
    labels_df = pd.DataFrame({'GAMEID':game_IDs, 'LABEL': labels_col})

    print('Saving labels')
    labels_df.to_csv('labels_key.csv', index = False)

    #make_TFRec_unpack(packed_data[0])
    print('Processing')

    # Save Locally
    if not g_project:
        # Where to put them
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # Get the game list
        num_games = len(game_IDs)
        source_dirs = [source_dir]*num_games
        save_dirs = [save_dir]*num_games
        hyp_args = [hyp_arg]*num_games



        # Package it for the pool
        packed_data = zip(game_IDs,source_dirs,save_dirs,hyp_args,[labels_dict]*len(game_IDs))

        pool = mp.Pool(processes = num_slaves)
        pool.map(make_TFRec_unpack,packed_data)

    # Save in a cloud bucket
    else:
        from google.cloud import storage
        # Conncet to google bucket
        client = storage.Client(project=g_project)
        bucket = client.get_bucket(g_bucket)

        # Where to put them
        save_dir = save_dir
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        # Get the game list
        game_IDs = os.listdir(source_dir)
        num_games = len(game_IDs)

        source_dirs = [source_dir]*num_games
        save_dirs = [save_dir]*num_games
        hyp_args = [hyp_arg]*num_games
        knocks = [knocks]*num_games
        
        # Package it for the pool
        packed_data = zip(game_IDs,source_dirs,save_dirs,hyp_args,knocks)


        #make_TFRec_cld_unpack(packed_data[0])
        #map(make_TFRec_cld_unpack,packed_data)
        pool = mp.Pool(processes = num_slaves)
        pool.map(make_TFRec_cld_unpack,packed_data)

        pool.close()
        pool.join()

        # Remove temporary directories
        tmps = os.listdir(save_dir)
        names = ['test','train','val']
        for dirp in tmps:
            for name in names:
                os.removedirs(save_dir+'/'+dirp+'/'+name)
            
        


    
        