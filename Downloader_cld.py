#!
#---------------------------------------------------------------------------
#
#       DOWNLOADS THE IMAGES FROM THE URLS FROM COMMUNITY
#
#---------------------------------------------------------------------------



import os
import numpy as np
import pandas as pd
import multiprocessing as mp
# request is for python3 not 2
#from urllib import request
from urllib2 import urlopen
from scipy import misc
from skimage import transform, io
import Top100Games_cld
import CommunityImages_cld


# Take a url and return an image
def get_images(url,ID,tag):
    try:
        # request is for python3
        #img = request.urlopen(url)
        if tag % 5 ==0:
            print(ID+': requesting image '+str(tag))
        img = urlopen(url)
        img = misc.imread(img, mode='RGB')

    except:
        print('Image not found')
    
    return img

# Save an image with unique tag
def img_saver(img_ID_tag):
    img,ID,tag = img_ID_tag
    img_path = 'GameImages/'+ID+'/'+ID+'_'+str(tag)+'.jpg'
    if tag % 5 ==0:
        print(ID+': saving image '+str(tag))
    io.imsave(img_path,img)

# Take a list of the form [(url,ID,ix)] and download the images to disk
def image_collector(url_ID_tag):
    b_size = len(url_ID_tag)
    images = [0]*b_size
    for i_im in range(b_size):
        url,ID,tag = url_ID_tag[i_im]
        img = get_images(url,ID,tag)
        images[i_im] = (img,ID,tag)
    
    map(img_saver,images)

# Break up a games urls into chunks
def url_chunks(ID):
    # Make an image directory if not already
    if not os.path.isdir('GameImages/'+ID):
        os.makedirs('GameImages/'+ID)

    # Get the csv as DF
    path = 'Gameurls/'+ID+'/'
    if not os.path.exists(path):
        print(ID+': Delete Gamesurl and run again')

    url_df = pd.read_csv(path+ID+'_urls.csv')

    # Only get urls that haven't been downloaded before
    url_list = url_df.loc[url_df['DOWNLOADED'] == 0.0,'URL']
    labels = [ID]*len(url_list)
    tag  = list(range(len(url_list)))

    # zip up urls with their ID and tag
    url_list = zip(url_list,labels,tag)

    
    size = 100

    chunks = len(url_list)/size
    orphans = len(url_list)%size

    url_blocks = [0]*chunks
    for i in range(chunks):
        url_blocks[i] = url_list[size*i:size*(i+1)]

    url_blocks.append(url_list[-orphans:])
    
    return url_blocks
    
if __name__ == '__main__':
    
    # Get the top100games list
    t100_df = pd.read_csv('Top100Games_cld.csv')

    # Check if there is an image directory
    if not os.path.isdir('GameImages'):
        os.makedirs('GameImages')

    # Find the url directories
    if not os.path.isdir('Gameurls'):
        CommunityImages_cld.main()
        
    game_dir = os.listdir('Gameurls')
    
    for game in game_dir:
        print('Starting '+game)
        url_blocks = url_chunks(game)
        # Find how many resources are available
        num_slaves = mp.cpu_count()-1
        pool = mp.Pool(processes = num_slaves)

        pool.map(image_collector,url_blocks)  
        print('Finished '+game)



    
