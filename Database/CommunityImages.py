#---------------------------------------------------------------------------------------------------------------------
#
#       GENERATE LIST OF IMAGE URL FOR TOP100GAMES
#       CREATES DIRECTORY WITH FOLDERS FOR EACH GAME CONTAING 
#       A CSV OF IMAGE URLS AND A FOLDER OF DOWNLOADED IMAGES
#
#---------------------------------------------------------------------------------------------------------------------

import multiprocessing as mp
from lxml import html
from lxml import etree
import requests
import os
import Top100Games
import pandas as pd
import argparse

# A function that downloads urls from a game's community page
def get_urls(appID,Num_scrolls,UrlDir):
    """
    Creates a csv file with the url's of images on the appID's community page. 
    Inputs:
        appID: int, steam appID of game.
        Num_scrolls: int, Number of times to scroll through the community page.
        UrlDir: str, the directory to save the image urls
    Returns:
        None
    """
    game = str(appID)

    # Make a directory for each game to hold a text file of the urls
    game_url_path = UrlDir+'/'+game

    #Check if the game has a directory
    if not os.path.isdir(game_url_path):
        os.makedirs(game_url_path)

    # See if the csv has been made else make it
    if os.path.exists(game_url_path+'/'+game+'_urls.csv'):
        game_url_df = pd.read_csv(game_url_path+'/'+game+'_urls.csv')
    else:
        game_url_df = {'URL':[],'DOWNLOADED':[]}
        game_url_df = pd.DataFrame(game_url_df)

    # Scrape through with each scroll
    game_urls = []
    for scroll in range(1,Num_scrolls):
        pg = str(scroll)
        
        # Build the community url

        game_com = 'https://steamcommunity.com/app/'+game+'/homecontent/?userreviewsoffset=0&p='+pg
        game_com+= '&workshopitemspage='+pg+'&readytouseitemspage='+pg+'&mtxitemspage='+pg+'&itemspage='+pg+'&screenshotspage='+pg
        game_com+= '&videospage='+pg+'&artpage='+pg+'&allguidepage='+pg+'&webguidepage='+pg+'&integratedguidepage='+pg
        game_com+= '&discussionspage='+pg+'&numperpage=50&browsefilter=mostrecent&browsefilter=mostrecent&'
        game_com+= 'appid='+game+'&appHubSubSection=2&appHubSubSection=2&l=english&filterLanguage=default&searchText=&forceanon=1'
        # NOT ALL PAGES HAVE A COMMUNITY
        # See if the page exists:
        try:
            # Request the page
            if scroll % 5 == 0:
                print('%s: Requesting page %d'%(game, scroll))
            game_pg = requests.get(game_com)

            # Make the tree
            game_tree = html.fromstring(game_pg.content)

            #Scrape for the sources of the images displayed
            new_urls = game_tree.xpath('//img[@class = "apphub_CardContentPreviewImage"]/@src')
            game_urls += new_urls

        # IF the community does not exist
        except:
            print(game + ' community not found')
            game_img_urls = ['Community Not Found']
            
    # put the new urls together and get rid of repeats
    game_urls = list(set(game_urls))
    print('%s: Found %d new urls'%(game,len(game_urls)))

    found_urls_df =pd.DataFrame( {'URL': game_urls, 'DOWNLOADED': [0]*len(game_urls)})
    
    # append the new data        
    game_url_df = game_url_df.append(found_urls_df)

    #Drop Duplicates
    game_url_df.drop_duplicates(subset = 'URL')

    print(game+': Saving URLs')
    game_url_df.to_csv(game_url_path+'/'+game+'_urls.csv')

# Make an unpacked version
def get_urls_unpack(tag):
    """tag: list or tuple containing [appID,Num_scrolls,UrlDir]"""
    appID,Num_scrolls,UrlDir = tag
    get_urls(appID,Num_scrolls,UrlDir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument('--num-games',
        help='The number of games to download url lists for',
        default=98,
        type=int
    )

    parser.add_argument('--num-scrolls',
        help='The number of scrolls through the community pages. Each scroll is roughly 25-50 images',
        default=18,
        type=int
    )

    parser.add_argument('--num-cores',
        help='The number of cores to use.',
        default=mp.cpu_count(),
        type=int
    )

    parser.add_argument('--save-dir',
        help='The name of file directory to store the url lists',
        default='Gameurls',
        type=str
    )

    args = parser.parse_args()
        
#else:
    # Get the top games list
    if not os.path.exists('Top100Games.csv'):
        print('Creating Top 100 List...')
        Top100Games.main()

    Game_df = pd.read_csv('Top100Games.csv')

    # Make the Gameurl dir
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Get the appIDs
    appIDs = list(Game_df['STEAM ID'])[:args.num_games]

    # Package with other arguments
    for i in range(args.num_games):
        appIDs[i] = [appIDs[i],args.num_scrolls,args.save_dir]

    with mp.Pool(processes = args.num_cores) as pool:
        #pool = mp.Pool(processes = num_cores)
            #url_chunks = pool.map(block_image_collector,url_blocks)  
            pool.map(get_urls_unpack,appIDs)

    pool.close()
    pool.join()
    