
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
        url: (str), the url of an image to download.
        ID: (int), the Steam ID of the game. 
        tag: (int), a unique identifier for saving.
        knocks: (int), the number of requests to make for an image. 
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
            img = request.urlopen(url)

            if tag % 50 ==0 and knock==0:
                print(ID+': requesting image '+str(tag))
            
            # Here for python2
            #img = urlopen(url)
            img = misc.imread(img, mode='RGB')

            knock = knocks + 10
        
        # Catch exceptions
        except requests.exceptions.Timeout:
            print('%s: knocking' %game)
            knock+=1
            continue
        
        except requests.exceptions.ConnectionError:
            print('%s: knocking' %game)
            knock+=1
            continue

        
    if knock > knocks+2:
        return [img,0]
    else:
        print('%s: Image %s failed to download'%(ID,tag))
        return [0,1]


# Save an image with unique tag
def img_exp(img,ID,tag,sv_dir,cloud=False):
    """Gets images from a url and saves them to sv_dir/ID/ID_tag.jpg.
    Inputs:
        img: an image in a format which skimage can save. 
        ID: (int), the Steam ID for the game. 
        tag: (int), a unique identifier for saving.
        sv_dir: (str), the save directory.
        cloud: (bool), whether to send to GCP bucket.
    Returns:
        None
    """

    img_name = ID+'_'+str(tag)+'.jpg'
    img_name = '/'.join([sv_dir,ID,img_name])

    if tag % 50 ==0:
        print(ID+': saving image '+str(tag))
    
    # Upload to cloud
    if cloud:
        
        # make tmp img
        io.imsave(img_name,img)

        # make blob to upload
        blob = bucket.blob(img_name)

        # connection issues can get in the way so try a few times. 2 is usually plenty.
        success = False
        gknock = 0
        gknocks = 4
        
        while (not success and gknock < gknocks):
            try:
                blob.upload_from_filename(img_name)
                success = True
                if tag % 50 == 0:
                    print('%s: FINISHED Uploading %s-data in %d tries'%(ID,img_name,gknock+1))
            except:
                gknock+=1
            
        # Remove the local file
        if not success:
            print('%s: Failed to upload %s'%(ID,img_name))
        
        os.remove(img_name)

    else:
        io.imsave(img_name,img)

# Take (url,ID,ix,d) and download the image to disk
def image_collector(url,ID,tag,d,sv_dir,knocks,cloud=False):
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
            img_exp(img,ID,tag,sv_dir,cloud)
            return 1.0
        else:
            # These failed to download
            return 2.0
    else:
        # These were already downloaded
        return 1.0

# Unpacked version
def image_collector_unpacked(url_ID_tag_d_sv_dir_knocks_cld):
    url,ID,tag,d,sv_dir,knocks, cloud = url_ID_tag_d_sv_dir_knocks_cld
    return image_collector(url,ID,tag,d,sv_dir,knocks,cloud)

def block_image_collector(urls_IDs_tags_ds_sv_dir_cld):
    result = map(image_collector_unpacked,
                urls_IDs_tags_ds_sv_dir_cld)
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

    
    # GCLOUD Arguments
    parser.add_argument('--g-project',
        help='The name of your gcloud project',
        type=str
    )

    parser.add_argument('--g-bucket',
            help='The name of an existing gcloud bucket for upload. Required if G_CLOUD is set.',
            type=str
    )

    args = parser.parse_args()
        
    # Unpack local args
    t100_df = pd.read_csv(args.top100)

    save_dir, url_dir, size = args.save_dir, args.url_dir, args.chunk_size
    knocks = args.knocks
    num_cores = args.num_cores

    # Unpack cloud args
    g_project, g_bucket = args.g_project, args.g_bucket


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
        url_path = '/'.join( [url_dir,ID])

        if not os.path.exists(url_path):
            print(ID+': Delete Gamesurl and run again')

        url_df = pd.read_csv(url_path+'/'+ID+'_urls.csv')

        # Get the relevant info
        url_list = url_df['URL']
        
        num_urls = len(url_list)
        
        tags, downs = list(url_df.index), url_df['DOWNLOADED']
        
        
        if g_project:
            from google.cloud import storage
            
            cloud = True            

            # Conncet to google bucket
            client = storage.Client(project=g_project)
            bucket = client.get_bucket(g_bucket)

        else:
            cloud = False
        
        IDs, clouds, save_dirs, knockss = [game]*num_urls, [cloud]*num_urls, [save_dir]*num_urls, [knocks]*num_urls
        

        # Zip it up
        url_data = zip(url_list,IDs,tags,downs,save_dirs,knockss,clouds)
        url_data = list(url_data)

        chunks = int(len(url_data)/size)
        orphans = len(url_data)%size

        url_blocks = [0]*chunks
        for i in range(chunks):
            url_blocks[i] = url_data[i*size:(i+1)*size]
        
        if orphans != 0:
            url_blocks.append(url_data[-orphans:])
        
        
        with mp.Pool(processes = num_cores) as pool:
            url_chunks = pool.map(block_image_collector,url_blocks)  

        pool.close()
        pool.join()

        new_url_list = []
        for x in url_chunks:
            new_url_list+= x
        
        # update urllist
        url_df['DOWNLOADED'] = new_url_list
        url_df.to_csv(url_path+'/'+ID+'_urls.csv')
        print('Finished '+game)

        # Clean up temp cloud dir
        if g_project:
            os.rmdir(game_path)


    # Clean up temp cloud dir
    if g_project:
        os.rmdir(save_dir)



