#
#---------------------------------------------------------------------------
#
#       DOWNLOADS THE IMAGES FROM THE URLS FROM COMMUNITY
#
#---------------------------------------------------------------------------
import argparse
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
# request is for python3 not 2
from urllib import request
#from urllib2 import urlopen
from scipy import misc
from skimage import transform, io
import Top100Games
import CommunityImages
import socket


# Define a timeout variable for Requests
timeout = 0.5 # This is a 1 second timeout
socket.setdefaulttimeout(timeout)


# Take a url and return an image
def get_images(url,ID,tag,knocks):
    """
    Gets images from a url
    Inputs:
        url: str, the url of an image to download.
        ID: int, the Steam ID of the game. 
        tag: int, a unique identifier for saving.
        knocks: int, the number of requests to make for an image. 
    Returns:
        If image download fails: [0,1] (1 deontes faild image and is used to update url csv)
        If image sucessfully downloaded: [img,0]
        where img is a scipy.misc read image read as 'RGB'. (0 denots successfull download and is used to update url csv)
    """
    # Initailzie image requests
    knock = 0
    while knock <= knocks:
        try:
            # request is for python3
<<<<<<< HEAD:Database/Downloader.py
            #img = request.urlopen(url)
            img = urlopen(url,timeout=1)
            if tag % 50 ==0 and knock==0:
                print(ID+': requesting image '+str(tag))
            
=======
            img = request.urlopen(url)
            if tag % 50 ==0 and knock==0:
                print(ID+': requesting image '+str(tag))
            
            #img = urlopen(url)
>>>>>>> Fixed memory leak in Downloader:Downloader.py
            img = misc.imread(img, mode='RGB')

            knock = knocks + 10
        except:
            if knock > 2:
                print('%s: Image %s not found after %d knocks'%(ID,tag,knock+1))
                knock+=1
        
    if knock > knocks+2:
        return [img,0]
    else:
        print('%s: Image %s failed to download'%(ID,tag))
        return [0,1]


# Save an image with unique tag
def img_saver(img,ID,tag,sv_dir):
    """Gets images from a url and saves them to sv_dir/ID/ID_tag.jpg.
    Inputs:
        img: an image in a format which skimage can save. 
        ID: int, the Steam ID for the game. 
        tag: int, a unique identifier for saving.
        sv_dir: str, the save directory.
    Returns:
        None
    """
    img_path = sv_dir+'/'+ID+'/'+ID+'_'+str(tag)+'.jpg'
    if tag % 50 ==0:
        print(ID+': saving image '+str(tag))
    io.imsave(img_path,img)

# Take (url,ID,ix,d) and download the image to disk
def image_collector(url,ID,tag,d,sv_dir,knocks):
    """Collects images from a url and saves them to sv_dir/ID/ID_tag.jpg. But first checks their download status
    Inputs:
        img: an image in a format which skimage can save. 
        ID: int, the Steam ID for the game. 
        tag: int, a unique identifier for saving.
        d: [0.0,1.0,2.0],
            0.0: Url has no download attempts. 
            1.0: Url has already been successfully downloaded.
            2.0: Url failed to download.
        sv_dir: str, the save directory.
    Returns:
        if d not 0.0 returns d. If d is zero returns 0.0 or 1.0 if download succeeds or fails respectively.
    """
    
    if d == 0.0:
        # These have no download attempts
        img, stat = get_images(url,ID,tag,knocks)
        if stat == 0:
            # These were downloaded sucessfully
            img_saver(img,ID,tag,sv_dir)
            return 1.0
        else:
            # These failed to download
            return 2.0
    else:
        # These were already downloaded
        return 1.0

# Unpacked version
def image_collector_unpacked(url_ID_tag_d_sv_dir_knocks):
    url,ID,tag,d,sv_dir,knocks = url_ID_tag_d_sv_dir_knocks
    return image_collector(url,ID,tag,d,sv_dir,knocks)

def block_image_collector(urls_IDs_tags_ds_sv_dir):
    result = map(image_collector_unpacked,
                urls_IDs_tags_ds_sv_dir)
    return list(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument('--knocks',
        help='The number of attemps to request an image',
        default=4,
        type=int
    )

    parser.add_argument('--num-cores',
        help='The number of cores to use.',
        default=mp.cpu_count(),
        type=int
    )

    parser.add_argument('--save-dir',
        help='The name of file directory to store the images',
        default='GameImages',
        type=str
    )

    parser.add_argument('--url-dir',
        help='The name of directory with the urls',
        default='Gameurls',
        type=str
    )
    # add to community images
    parser.add_argument('--top100',
        help='The name of Top 100 Games csv',
        default='Top100Games.csv',
        type=str
    )
    
    parser.add_argument('--chunk-size',
        help='The size of chunks to break up image download into',
        default=100,
        type=int
    )

    args = parser.parse_args()
    
    # Find how many resources are available
    num_slaves = mp.cpu_count()
    
    # Unpack args
    t100_df = pd.read_csv(args.top100)
    save_dir = args.save_dir
    url_dir = args.url_dir
    size = args.chunk_size
    knocks = args.knocks
    num_cores = args.num_cores


    # Check if there is an image directory
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Find the url directories
    if not os.path.isdir(url_dir):
        CommunityImages.main()
        
    game_dir = os.listdir(url_dir)
    
    # Loop through games
    for game in game_dir:
        print('Starting '+game)
        ID = game

        # Where the images will be saved
        game_path = save_dir+'/'+ID

        # Make an image directory if not already
        if not os.path.isdir(game_path):
            os.makedirs(game_path)

        # Get the csv as DF
        url_path = url_dir+'/'+ID+'/'
        if not os.path.exists(url_path):
            print(ID+': Delete Gamesurl and run again')

        url_df = pd.read_csv(url_path+'/'+ID+'_urls.csv')

        # Get the relevant info
        url_list = url_df['URL']
        downs = url_df['DOWNLOADED']
        IDs = [game]*len(url_list)
        tags = list(url_df.index)

        # Zip it up
        url_data = zip(url_list,IDs,tags,downs,[save_dir]*len(url_list),[knocks]*len(url_list))
        url_data = list(url_data)

<<<<<<< HEAD:Database/Downloader.py
=======
        url_data = list(url_data)
>>>>>>> Fixed memory leak in Downloader:Downloader.py
        chunks = int(len(url_data)/size)
        orphans = len(url_data)%size

        url_blocks = [0]*chunks
        for i in range(chunks):
            url_blocks[i] = url_data[i*size:(i+1)*size]
        
        if orphans != 0:
            url_blocks.append(url_data[-orphans:])
        
        # Find how many resources are available
        with mp.Pool(processes = num_cores) as pool:
        #pool = mp.Pool(processes = num_cores)
            url_chunks = pool.map(block_image_collector,url_blocks)  

        pool.close()
        pool.join()

        new_url_list = []
        for x in url_chunks:
            new_url_list+= x
        
        # update urllist
        url_df['DOWNLOADED'] = new_url_list
        url_df.to_csv(game_path+'/'+ID+'_urls.csv')
        print('Finished '+game)
