#----------------------------------------------------------------------------------------
#
#       PROCESS THE IMAGES PROCURED BY CommunityImages.py
#       RETURN TENSOR OF PROCESSED IMAGES IN GAME FOLDER
#
#----------------------------------------------------------------------------------------


import os
import numpy as np
from scipy import misc
from scipy import ndimage 
import CommunityImages
import pandas as pd
import matplotlib.pyplot as plt


# If "rawimages" folder exists ImagProcessor.py crawls through the game folders inside "rawimages". Otherwise CommunityImages.py is called.
# For each game folder it goes through the "downloads" folder and resizes all the images.
# The resized images are collected in the rank 4 tensor g_proc_images which is saved as a numpy bit file in the game folder.



# Set the desired resolution
resolution = (64,64)

# Check if the rawimage file exists
if not(os.path.isdir('rawimages')):
    print('rawimages file not found')
    print('Running CommunityImages')
    CommunityImages.main()
    
else:
    print('Found rawimages file')

# Get a list of the game file names

games_folders = os.listdir('rawimages/')

n_fold = len(games_folders)

# The total raw data will be here

all_processed = [0]*n_fold

# Make the labels data
labels = [0]*n_fold

g_df = pd.read_csv('Top100Games.csv')

# Loop through the games and make processed folders

#Count total images
total_images = 0
for ifol in range(n_fold):
    g_name = games_folders[ifol]

    # Get gameID
    g_ix = g_df[g_df['GAME'] == g_name].index[0]
    g_ID = g_df['STEAM ID'].iloc[g_ix]
    labels[ifol]
   

    

    g_folder = 'rawimages/'+g_name
    g_image_folder = g_folder+'/downloads'

    # Some files don't have images 
    if not os.path.isdir(g_image_folder):
        CommunityImages.main()

    # Get the names of image files
    g_images = os.listdir(g_image_folder)

    num_imgs = len(g_images)


    # Create a list of images
    g_proc_images = np.zeros((len(g_images), resolution[0], resolution[1],3))

    
    for ig in range(num_imgs):
        total_images += 1

        image = g_images[ig]
        img_path = g_image_folder+'/'+image
#        print(img_path)
        
        img_array = ndimage.imread(img_path)

        # Change the resolution
        img_array = misc.imresize(img_array, resolution)

        # Add to the games images tensor
        g_proc_images[ig] = img_array
    
    # Make it an array
    g_proc_images = np.array(g_proc_images)
    
    #print(g_proc_images.shape)

    # Add the array to the 
    all_processed[ifol] = g_proc_images
    
    # update labels
    labels[ifol] = [g_ID]*num_imgs    




# Make processed data and labels an array
all_processed = np.array(all_processed)
labels = np.array(labels)

# Unwind them
all_processed.shape = (total_images, resolution[0],resolution[1],3)
labels.shape = total_images

# Save them as numpy arrays
if not os.path.isdir('processed/'):
    os.makedirs('processed/')
np.save('processed/'+'images',all_processed)
np.save('processed/'+'labels', labels)




        
    