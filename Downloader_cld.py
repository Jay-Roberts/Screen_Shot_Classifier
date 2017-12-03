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

# Set resolution
resolution = (224,224,3)
# Set a batch size for saving
batch = 100

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

    k = len(url_list)%batch
    batches = int((len(url_list)-k)/100)
    for i in range(batches):
        start,stop = i*batches,(i+1)*batches

        # Hold image files to be saved later
        images = [0]*(stop-start)
        for ix in range(start,stop):
            url = url_list[ix]

            #Net issues could stop this
            try:
                img = request.urlopen(url)
                img = misc.imread(img, mode='RGB')

                # Process the image
                img = transform.resize(img,resolution)

                # Put it in the working image list
                # also store its index for unique identifier
                images[ix-start] = [img,ix]
            except:
                print('Image not found')
        
        # Save the images this will change in real cloud version but for testing it saves locally
        print('Saving Images')
        for imgx  in range(len(images)):
            # Only grab the images that went through
            if not images[imgx] ==0:
                # Save the image
                img, tag = images[imgx]
                img_path = 'GameImages/'+ID+'/'+ID+str(tag)+'.jpg'
                io.imsave(img_path,img)

                # Update the download column
                url_df.iloc[tag]['DOWNLOADED'] = 1
    
    # Save the updated dataframe with downloaded markers
    url_df.to_csv(path+ID+'_urls.csv', index = False)







    
