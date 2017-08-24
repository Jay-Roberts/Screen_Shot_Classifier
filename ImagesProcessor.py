import os
import pandas as pd
import numpy as np
from scipy import misc
from scipy import ndimage 
import CommunityImages
import matplotlib.pyplot as plt



#stuff2 = ndimage.imread('/home/jay/Desktop/Codes/WebScrapping/OpenCVPractice/Dota_2_URLsDota_24')

#stuff4 = misc.imresize(stuff2, (124,512))
#misc.imshow(stuff4)

resolution = (512,256)

# Check if the rawimage file exists
if not(os.path.isdir('rawimages')):
    print('rawimages file not found')
    print('Running CommunityImages')
    CommunityImages.main()
    
else:
    print('Found rawimages file')

# Get a list of the game file names

games_folders = os.listdir('rawimages/')
print(games_folders)

# Loop through the games and make processed folders

for g_name in games_folders:
    g_folder = 'rawimages/'+g_name
    g_image_folder = g_folder+'/downloads'

    # Create the processed folder

    g_proc_folder = 'rawimages/'+g_name+'/processed'

    if not(os.path.isdir(g_proc_folder)):
        os.makedirs(g_proc_folder)
    
    # Some files don't have images 
    if os.path.isdir(g_image_folder):

        # Get the names of image files
        g_images = os.listdir(g_image_folder)

        num_imgs = len(g_images)
        # Create a list of images
        g_proc_images = np.zeros((len(g_images), resolution[0], resolution[1],3))

        for ig in range(num_imgs):
            image = g_images[ig]
            img_path = g_image_folder+'/'+image
            #print(img_path)
            
            img_array = ndimage.imread(img_path)

            # Change the resolution
            img_array = misc.imresize(img_array, resolution)

            g_proc_images[ig] = img_array
            g_proc_images = np.array(g_proc_images)
        

            #img_array1 = ndimage.median_filter(img_array, size = 1)
        np.save(g_folder+'/'+g_name,g_proc_images)
#misc.imshow(img_array)
#misc.imshow(img_array1)




        
    