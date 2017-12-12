#---------------------------------------------------------------------------
#
#       DOWNLOADS THE IMAGES FROM THE URLS FROM COMMUNITY
#
#---------------------------------------------------------------------------



import os
import numpy as np
import pandas as pd
# request is for python3 not 2
#from urllib import request
from urllib2 import urlopen
from scipy import misc
from skimage import transform, io
import Top100Games_cld
import CommunityImages_cld

def main(resolution = (224,224,3), all = True, begin = 0, end = 100, save = True):
    """
    INPUTS:{resolution: the resolution of the processed image, first: index of first image to be processed (inclusive)
    last: index of last image to be processed (not inclusive), save: if true saves the images locally}
    RETURNS: List of processed images from index first to last. Images stored in numpy arrays of size resolution.
    """
    # Get the top100games list
    t100_df = pd.read_csv('Top100Games_cld.csv')

    # Check if there is an image directory
    if not os.path.isdir('GameImages'):
        os.makedirs('GameImages')

    # Find the url directories
    if not os.path.isdir('Gameurls'):
        CommunityImages_cld.main()
        
    game_dir = os.listdir('Gameurls')

    # Loop through each directory
    for ID in game_dir:
        print(ID)

        # Make an image directory if not already
        if not os.path.isdir('GameImages/'+ID):
            os.makedirs('GameImages/'+ID)

        # Get the csv as DF
        path = 'Gameurls/'+ID+'/'
        if not os.path.isfile(path):
            print('Delete Gamesurl and then run again')
        
        url_df = pd.read_csv(path+ID+'_urls.csv')
        # Only get urls that haven't been downloaded before
        url_list = url_df.loc[url_df['DOWNLOADED'] == 0.0,'URL']
        
        # Hold image files to be saved later
        if all:
            first = 0
            last = len(url_list)
        else:
            first = begin
            last = end
        
        images = [0]*(last -first)
        print('Downloading: %d images'%(last-first))
        for ix in range(first,last):
            url = url_list[ix]

            #Net issues could stop this
            try:
                # request is for python3
                #img = request.urlopen(url)
                img = urlopen(url)
                img = misc.imread(img, mode='RGB')

                if ix % 50 == 0:
                    print('Processing image %d/%d'%(ix,num_imgs))
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
            num_imgs = len(images)
            for imgx  in range(num_imgs):

                # Only grab the images that went through
                if not images[imgx] ==0:
                    # Save the image
                    img, tag = images[imgx]
                    img_path = 'GameImages/'+ID+'/'+ID+'_'+str(tag)+'.jpg'
                    io.imsave(img_path,img)

                    # Update the download column
                    url_df.iloc[tag]['DOWNLOADED'] = 1

                    if ix % 50 == 0:
                        print('Saving image %d/%d'%(tag,num_imgs))
            
            # Update the Top100games     
            print('Updating csvs')
            t100_df.loc[t100_df['STEAM ID'] == int(ID), 'IMAGES'] += num_imgs
            t100_df.to_csv('Top100Games_cld.csv')
            url_df.to_csv(path+ID+'_urls.csv')
        else:
            return images

if __name__ == '__main__':
    main()


    
