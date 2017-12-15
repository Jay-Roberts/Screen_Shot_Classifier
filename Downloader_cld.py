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
        if tag % 50 ==0:
            print(ID+': requesting image '+str(tag))
        img = urlopen(url)
        img = misc.imread(img, mode='RGB')

        return [img,0]
    except:
        print('Image not found')
        return [0,1]

# Save an image with unique tag
def img_saver(img_ID_tag):
    img,ID,tag = img_ID_tag
    img_path = 'GameImages/'+ID+'/'+ID+'_'+str(tag)+'.jpg'
    if tag % 50 ==0:
        print(ID+': saving image '+str(tag))
    io.imsave(img_path,img)

# Take (url,ID,ix,d) and download the image to disk
def image_collector(url_ID_tag_d):
    url,ID,tag,d = url_ID_tag_d

    if d == 0.0:
        img, stat = get_images(url,ID,tag)
        if stat == 0:
            img_ID_tag= (img,ID,tag)
            img_saver(img_ID_tag)
            return 1.0
        else:
            return 2.0
    else:
        return 1.0

def block_image_collector(urls_IDs_tags_ds):
    result = map(image_collector,urls_IDs_tags_ds)
    return list(result)

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
        ID = game

        # Make an image directory if not already
        if not os.path.isdir('GameImages/'+ID):
            os.makedirs('GameImages/'+ID)

        # Get the csv as DF
        path = 'Gameurls/'+ID+'/'
        if not os.path.exists(path):
            print(ID+': Delete Gamesurl and run again')

        url_df = pd.read_csv(path+ID+'_urls.csv')

        # Get the relevant info
        url_list = url_df['URL']
        downs = url_df['DOWNLOADED']
        IDs = [game]*len(url_list)
        tags = list(url_df.index)

        # Zip it up
        url_data = zip(url_list,IDs,tags,downs)

        size = 100
        chunks = len(url_data)/size
        orphans = len(url_data)%size

        url_blocks = [0]*chunks
        for i in range(chunks):
            url_blocks[i] = url_data[i*size:(i+1)*size]
        
        if orphans != 0:
            url_blocks.append(url_data[-orphans:])
        
        # Find how many resources are available
        num_slaves = mp.cpu_count() - 1
        pool = mp.Pool(processes = num_slaves)
        url_chunks = pool.map(block_image_collector,url_blocks)  

        new_url_list = []
        for x in url_chunks:
            new_url_list+= x
        
        # update urllist
        url_df['DOWNLOADED'] = new_url_list
        url_df.to_csv(path+ID+'_urls.csv')
        print('Finished '+game)