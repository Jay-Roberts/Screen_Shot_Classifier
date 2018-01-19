import tensorflow as tf
import numpy as np
import argparse
import os
import cv2

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


# Feed iamges as np.arrays into the trained model
def predict_np(images,graph_path):
    """
    Takes a numpy image and uses the model from graph_path to predict what game it is.
    Inputs:
        images: (list), numpy arrays to be classified. Must fit the resolution for the graph.
        graph_path: (str), relative path to the trained model .meta files.
    Returns:
        A list [inference,coinfidence].
            inference: A list of int The softmax guess corresponding to the inferred label for the image.
            confidence: A list where each element is a list with the probability the image corresponds to a given label.
    """
    # Init graph
    graph = tf.Graph()
    
    # Init session
    with tf.Session(graph=graph) as sess:

        # Load the graph with the trained states
        loader = tf.train.import_meta_graph(graph_path + '.meta')
        loader.restore(sess, graph_path)

        print('Neural Network Successfully Loaded')

        # Get the input layer
        input = graph.get_operation_by_name("input_layer").outputs[0]
        input_dict = {input: images}

        # Get the predicted class along with the confidence of all possibilities
        prediction = graph.get_operation_by_name("softmax_tensor").outputs[0]
        classes = graph.get_tensor_by_name("class:0")
        

        # Feed in data to be predicted
        sess.run([prediction,classes],feed_dict=input_dict)

        # Find the probabilities of image being each game
        confidence = prediction.eval(feed_dict=input_dict,session=sess)

        # Get actual guess
        inference = classes.eval(feed_dict=input_dict,session=sess)

        return [inference,confidence]


# Take a list of img file names and return
def predict_imgs(images,graph_path,res=(28,28)):

    num_imgs = len(images)
    # Get file names w/o dir
    names = [img_path.split('/')[-1] for img_path in images]
    # Get file names w/o extension
    names = [name.split('.')[0] for name in names]

    # Init input
    input_array = np.zeros((num_imgs,28,28,3))

    # Fill input
    for img_ix in range(num_imgs):
        img_path = images[img_ix]
        name = names[img_ix]

        img = load_image(img_path,res)
        input_array[img_ix,:,:,:] = img
    
    predictions = predict_np(input_array,graph_path)
    predictions.append(names)
    
    return predictions








if __name__ == '__main__':

    # Used for scanning for latest checkpoint, which is passed as a command line argument
    parser = argparse.ArgumentParser(description='Load the latest Saved trained Nueral Netowrk')

    parser.add_argument('-d', 
                        '--ckpt-file', 
                        metavar='REL_PATH',
                        type=str,
                        help='The relative path for the checkpoint file')
    
    parser.add_argument('--img-dir', 
                        metavar='IMG_PATH',
                        type=str,
                        help='The path for the images directory',
                        default = 'Testimages')
                        
    parser.add_argument('--res', 
                        metavar='RES',
                        type=tuple,
                        help='The resolution required for inference',
                        default = (28,28,3))

    # Parse them
    args = parser.parse_args()

    # Get path
    graph_path = os.getcwd() + args.ckpt_file
    print('Graph path: %s'%(graph_path))
    
    # testing. Delete later
    Z = np.zeros((28,28,3))
    I = np.array([np.identity(28)]*3)
    I.shape = (28,28,3)
    newimages = [Z,I]
    print(predict_np(newimages,graph_path))

    # Get image directory and image path names
    img_dir = args.img_dir
    images = os.listdir(img_dir)
    images = ['/'.join([img_dir,img]) for img in images]

    print(predict_imgs(images,graph_path))


