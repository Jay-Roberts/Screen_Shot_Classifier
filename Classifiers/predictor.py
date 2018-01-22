import tensorflow as tf
import numpy as np
import argparse
import os
import cv2
import pandas as pd

# Loads images
def load_image(addr,res):
    """
    Loads an image from a file as a float32 numpy array.It resizes to res and normalizes data.
    Inputs:
        addr: The file path to the image file. (str)
        res: Desired resolution. (list)
    Returns:
        Resized image as numpy array shape (res,3) with dtype np.float32 in range [-0.5,0.5]
        to fit with normalized input from training data.
    """
    res = tuple(res)
    # cv2 loads images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img,res)
    img.astype(np.float32)
    # Normalize the values of the image from the range [0, 255] to [-0.5, 0.5]
    img = img / 255 - 0.5    
    return img


# Feed images as np.arrays into the trained model
def predict_np(images,graph_path):
    """
    Takes a numpy image and uses the model from graph_path to predict what label it is. Returns a
    dictionary of infered games and the confidence probabilities.
    Inputs:
        images: (list), list of numpy array to be classified. Must fit the resolution for the graph.
        graph_path: (str), relative path to the trained model .pb file.
    Returns:
        Dict with keys {inferences,confidences}.
            inferences: int, The softmax guess corresponding to the inferred label for the image.
            confidences: A list where each element is a list with the probability the image corresponds to a given label.
    """
    
    # Use the predictor loader to load from .pb files
    saved_model_predictor = tf.contrib.predictor.from_saved_model(export_dir=graph_path)
    
    num_imgs =images.shape[0]

    inferences, confidences = [0]*num_imgs,[0]*num_imgs
    for j in range(num_imgs):
        # Get image
        img = images[j,:,:,:]
        # Predictor HUNGRY for dictionary with one image
        input_dict = {'image': img}
        # Predict it
        output_dict =saved_model_predictor(input_dict)

        inferences[j], confidences[j]= list(output_dict["classes"]), list(output_dict["probabilities"][0])

    # Make return dictionary
    results = {'inferences':inferences,'confidences': confidences}
    return results
    

# Take a list of img file names and returns predictions
def predict_imgs(images,graph_path,res=(28,28)):
    """
    Takes list of img file names and returns dictionary of infrered label and confidence probabilites.
    Inputs:
        images: list, file names of image as str.
        graph_path: str, directory contaning the saved .pb model file.
        res: tuple, resolution of image required for the model to predict.
    Returns:
        Dict with keys {names,inferences,confidences}.
            names: str, name of file wihtout directory tree or file extension. 
            inferences: int, The softmax guess corresponding to the inferred label for the image.
            confidences: A list where each element is a list with the probability the image corresponds to a given label.    
    """

    num_imgs = len(images)
    # Get file names w/o dir
    names = [img_path.split('/')[-1] for img_path in images]
    # Get file names w/o extension
    names = [[name.split('.')[0]] for name in names]

    # Init input
    input_array = np.zeros((num_imgs,28,28,3))

    # Fill input
    for img_ix in range(num_imgs):
        img_path = images[img_ix]

        img = load_image(img_path,res)
        input_array[img_ix,:,:,:] = img
    
    results = predict_np(input_array,graph_path)
    results['names'] = names

    return results

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
                        default = 'estimator/Testimages')

    parser.add_argument('--save', 
                        metavar='SV_PATH',
                        type=str,
                        help='Where to save the csv of predictions',
                        default = 'inferences')
                        
    parser.add_argument('--res', 
                        metavar='RES',
                        type=tuple,
                        help='The resolution required for inference',
                        default = (28,28,3))

    # Parse them
    args = parser.parse_args()
    
    # Get path
    wrk_dir = os.getcwd()
    graph_path = '/'.join([wrk_dir,args.ckpt_file])
    print('Graph path: %s'%(graph_path))

    # Get image directory and image path names
    img_dir = args.img_dir
    images = os.listdir(img_dir)
    images = ['/'.join([img_dir,img]) for img in images]

    num_imgs = len(images)

    # Get the labes to GAMEID
    labels_key = pd.read_csv('labels_key.csv')
    games = list(labels_key['GAMEID'])

    
    # Get prediction results
    results =predict_imgs(images,graph_path)

    # Format them to be a DataFrame
    results = list(zip(results['names'],results['confidences'],results['inferences']))

    for rix in range(len(results)):
        name, conf, infer = results[rix]
        # Switch inference from label to gameID
        infer = [games[infer[0]]]
        results[rix] = name+conf+infer

    columns = ['image']
    columns+= games
    columns+= ['inference']

    results_df = pd.DataFrame(results, columns=columns)
    # Round the probabilities 
    results_df[games] = results_df[games].apply(lambda x: pd.Series.round(x,4))

    # Save the results
    results_df.to_csv(args.save+'.csv')




    

