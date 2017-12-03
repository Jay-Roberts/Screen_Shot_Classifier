#---------------------------------------------------------------------------
#
#       DOWNLOADS THE IMAGES FROM THE URLS FROM COMMUNITY
#
#---------------------------------------------------------------------------



import os
import numpy as np
import pandas as pd
from urllib import request
from scipy import misc
from skimage import transform, io
import Top100Games_cld

def main(resolution = (224,224,3),first = 0, last = 100, save = False):
    """
    Inputs;{resolution: the resolution of the processed image, first: index of first image to be processed (inclusive)
    last: index of last image to be processed (not inclusive), save: if true saves the images locally}"""
    # Check if there is an image directory
    if not os.path.isdir('GameImages'):
        os.makedirs('GameImages')

    # Find the url directories
    game_dir = os.listdir('Gameurls')

    # Loop through each directory
    for ID in game_dir:
        print(ID)

        # Make an image directory if not already
        if not os.path.isdir('GameImages/'+ID):
            os.makedirs('GameImages/'+ID)

        # Get the csv as DF
        path = 'Gameurls/'+ID+'/'
        url_df = pd.read_csv(path+ID+'_urls.csv')
        url_list = url_df['URL']

        # Hold image files to be saved later
        images = [0]*(last -first)
        for ix in range(first,last):
            url = url_list[ix]

            #Net issues could stop this
            try:
                img = request.urlopen(url)
                img = misc.imread(img, mode='RGB')

                # Process the image
                img = transform.resize(img,resolution)

                # Put it in the working image list
                # also store its index for unique identifier
                images[ix-first] = [img,ix]

            except:
                print('Image not found')


        if save:
            # Save the images this will change in real cloud version but for testing it saves locally
            print('Saving Images')
            for imgx  in range(len(images)):
                # Only grab the images that went through
                if not images[imgx] ==0:
                    # Save the image
                    img, tag = images[imgx]
                    img_path = 'GameImages/'+ID+'/'+ID+'_'+str(tag)+'.jpg'
                    io.imsave(img_path,img)

                    # Update the download column
                    url_df.iloc[tag]['DOWNLOADED'] = 1
        else:
            return images




    