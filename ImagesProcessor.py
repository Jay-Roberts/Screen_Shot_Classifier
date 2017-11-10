#----------------------------------------------------------------------------------------
#
#       PROCESS THE IMAGES PROCURED BY CommunityImages.py
#       RETURN TENSOR OF PROCESSED IMAGES IN GAME FOLDER
#
#----------------------------------------------------------------------------------------
"""
- If "rawimages" folder exists ImagProcessor.py crawls through the game folders inside "rawimages". 
Otherwise CommunityImages.py is called.
- For each game the program goes through 'game_folder/downloads' and resizes the images.
- Resized images are stored in 'game_folder/processed' labeled by gameID_download_number.
- Images are then stored in the all_processed tensor, which has shape (Total number of images, (resolution,3)),
 alongside a labels tensor that stores the APPID and has shape (Total number of images). These tensors can
 be directly used as data-label pairs for training. Specifically
 image all_processed[i] has GameID labels[i].
"""

import os
import numpy as np
from scipy import misc
from scipy import ndimage 
import CommunityImages
import pandas as pd
import matplotlib.pyplot as plt

# Make the save directory
if not os.path.isdir('processed/'):
    os.makedirs('processed/')

# Set the desired resolution
resolution = (28,28)

# Check if the rawimage file exists
if not(os.path.isdir('rawimages')):
    print('rawimages file not found')
    print('Running CommunityImages')

    # USING DEFAULT NUM ARGUMENT [5,2,10]!
    CommunityImages.main(os_in = False)
    
else:
    print('Found rawimages file')

# Get a list of the game folder names
# Game Folders are named by AppID
games_folders = os.listdir('rawimages/')
n_fold = len(games_folders)

# Get the game data
g_df = pd.read_csv('Top100Games.csv')

# Make the data and labels tensor
all_processed = [0]*n_fold
labels = [0]*n_fold

# Columns for a data-frame organized by game
appid_col  = [0]*n_fold
images_col = [0]*n_fold

# Loop through the games
# Count total images
total_images = 0
for ifol in range(n_fold):
    
    g_name = games_folders[ifol]

    # Get gameID
    g_ix = ifol
    g_ID = g_df['STEAM ID'].iloc[ifol]
    
    # Find image folder
    g_folder = 'rawimages/'+g_name
    g_image_folder = g_folder+'/downloads'

    # Make folder to store the game's processed images for comparison sake
    if not os.path.isdir(g_folder+'/processed'):
        os.makedirs(g_folder+'/processed')

    # Some files don't have images.
    # Currently runs the WHOLE Community images. Change this to download only the missing games
    # Maybe keep the number of games in the download folder in the Top100Games.csv  
    #if not os.path.isdir(g_image_folder):
    #    CommunityImages.main()

    # Get the names of image files
    g_images = os.listdir(g_image_folder)

    # Set number of images
    num_imgs = len(g_images)

    # Create the tensor to hold the game's image arrays
    g_proc_images = np.zeros((len(g_images), resolution[0], resolution[1],3))
    
    # Loop through the images
    for ig in range(num_imgs):
        # Get image and make a file path
        image = g_images[ig]
        img_path = g_image_folder+'/'+image
        
        # Read picture
        img_array = ndimage.imread(img_path)

        #
        # Here is where we can add any other preprocessing we like
        #

        # Change the resolution
        img_array = misc.imresize(img_array, resolution)

        # Update g_proc_images
        g_proc_images[ig] = img_array

        # Save the processed image for future inspection
        misc.imsave(g_folder+'/processed/%d_%d.jpg' % (g_ID,ig), img_array)

        # Count
        total_images += 1
        
    # Make it an array
    g_proc_images = np.array(g_proc_images)
    
    

    # Update all_processed 
    all_processed[ifol] = g_proc_images
    
    # Update labels, appid and images
    labels[ifol] = [g_ID]*num_imgs    
    appid_col[ifol], images_col[ifol] = g_ID, g_proc_images

# Make all_processed and labels into arrays
all_processed = np.concatenate(all_processed, axis = 0)
labels = np.concatenate(labels, axis = 0)

# Unwind them
all_processed.shape = (total_images, resolution[0],resolution[1],3)
labels.shape = (total_images)

# Save the data
np.save('processed/'+'proc_imgs_tensor',all_processed)
np.save('processed/'+'proc_labels_tensor', labels)




        
    