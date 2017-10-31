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
# Images are stored in the all_processed tensor, which has shape (Total number of images, (resolution,3)), alongside a 
## labels tensor that stores the APPID and has shape (Total number of images).
# The images and labels are created at the same time so that the image all_processed[i] has GameID labels[i].
 




# Set the desired resolution
resolution = (28,28)

# Check if the rawimage file exists
if not(os.path.isdir('rawimages')):
    print('rawimages file not found')
    print('Running CommunityImages', '\n')
    CommunityImages.main()
    
else:
    print('Found rawimages file', '\n')

# Get a list of the game file names

games_folders = os.listdir('rawimages/')
n_fold = len(games_folders)


# Get the game1D data
g_df = pd.read_csv('Top100Games.csv')
#print(g_df.head())

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
    
    g_fol_name = games_folders[ifol]

    # Get gameID
    g_ix = ifol
    
    #g_ix = g_df[g_df['GAME'] == g_name].index[0]

    g_ID = g_df['STEAM ID'].iloc[ifol]
    g_name = g_df['GAME'].iloc[ifol]
    
    
    # Find image folder
    g_folder = 'rawimages/'+g_fol_name
    g_image_folder = g_folder+'/downloads'

    # Some files don't have images 
    if not os.path.isdir(g_image_folder):
        CommunityImages.main()

    # Get the names of image files
    g_images = os.listdir(g_image_folder)

    

    # Set number of images
    num_imgs = len(g_images)

    # Create the tensor to hold g_name's image arrays
    g_proc_images = np.zeros((len(g_images), resolution[0], resolution[1],3))
    
    print('Processing: ', g_name)
    print('Number of images: ', num_imgs, '\n')
    
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

        # Count
        total_images += 1
        

        # Make it an array
    g_proc_images = np.array(g_proc_images)
    #print(g_proc_images.shape)

    # Update all_processed 
    all_processed[ifol] = g_proc_images
    
    # Update labels, appid and images
    labels[ifol] = [g_ID]*num_imgs    
    appid_col[ifol], images_col[ifol] = g_ID, g_proc_images




# Make the save directory
if not os.path.isdir('processed/'):
    print('Creating Processed Folder','\n')
    os.makedirs('processed/')

# Make a data frame organized by game
# Save the data frame
#processed_images = pd.DataFrame({'IMAGES': images_col, 'APPID': appid_col})
#processed_images.to_pickle('processed/processed_images')



# Make all_processed and labels into arrays

all_processed = np.concatenate(all_processed, axis = 0)
labels = np.concatenate(labels, axis = 0)



# Unwind them
all_processed.shape = (total_images, resolution[0],resolution[1],3)
labels.shape = (total_images)

# Shuffle them together

from sklearn.utils import shuffle

print('Shuffling Data')
all_processed, labels = shuffle(all_processed,labels, random_state = 10)

# Save the data
print('Saving Tensors')
np.save('processed/'+'images',all_processed)
np.save('processed/'+'labels', labels)




        
    