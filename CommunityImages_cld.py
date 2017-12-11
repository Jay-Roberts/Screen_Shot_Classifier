#---------------------------------------------------------------------------------------------------------------------
#
#       GENERATE LIST OF IMAGE URL FOR TOP100GAMES
#       RETURNS DIRECTORY WITH FOLDERS FOR EACH GAME CONTAING A CSV OF IMAGE URLS AND A FOLDER OF DOWNLOADED IMAGES
#
#---------------------------------------------------------------------------------------------------------------------

"""
- If Top100Games.csv does not exist Top100Games.py is called to create it.
- If they don't exist 'Gameurls' and 'Gameurls/game_name' directories are created 
for each game. 
- In 'Gameurls/game_name' a csv file containing the URLs of images from the games steam community is created/updated.
the csv has a DOWNLOADED column. 0 = not downloaded, 1 = downloaded, or error-message
 """

from lxml import html
from lxml import etree
import requests
import os
import Top100Games_cld
import pandas as pd
def main(Num_scrolls = 18, save = True):

    Num_scrolls_str = str(Num_scrolls)
    # Get the top Num_games info 

    if not os.path.exists('Top100Games_cld.csv'):
        print('Creating Top 100 List...')
        Game_df = Top100Games_cld.main(save = False)
        Game_df.to_csv('Top100Games_cld.csv', index = False)
    else:
        Game_df = pd.read_csv('Top100Games_cld.csv')
    if not os.path.isdir('Gameurls'):
        os.makedirs('Gameurls')

    for index,row in Game_df.iterrows():
        appID, name = int(row['STEAM ID']), row['GAME']
        game = str(appID)

        # Make a directory for each game to hold a text file of the urls
        game_url_path = 'Gameurls/'+game

        #Check if the game has a directory
        if not os.path.isdir(game_url_path):
            os.makedirs(game_url_path)

        # See if the csv has been made else make it
        if os.path.exists(game_url_path+'/'+game+'_urls.csv'):
            game_url_df = pd.read_csv(game_url_path+'/'+game+'_urls.csv')
        else:
            game_url_df = {'URL':[],'DOWNLOADED':[]}
            game_url_df = pd.DataFrame(game_url_df)

        print('Game: ', name)
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
                game_pg = requests.get(game_com)

                # Make the tree
                game_tree = html.fromstring(game_pg.content)

                #Scrape for the sources of the images displayed
                new_urls = game_tree.xpath('//img[@class = "apphub_CardContentPreviewImage"]/@src')
                game_urls += new_urls

            # IF the community does not exist
            except:
                print(name + ' community not found')
                game_img_urls = ['Community Not Found']
                
        # put the new urls together and get rid of repeats
        game_urls = list(set(game_urls))
        found_urls_df =pd.DataFrame( {'URL': game_urls, 'DOWNLOADED': [0]*len(game_urls)})
        print('Found %d new urls'%(len(game_urls)))

        # append the new data        
        game_url_df = game_url_df.append(found_urls_df)

        #Drop Duplicates
        game_url_df.drop_duplicates(subset = 'URL')

        if save:
            game_url_df.to_csv(game_url_path+'/'+game+'_urls.csv',index = False)
            print(len(game_url_df['URL']))
            print('Saving URLs')
        else:
            return game_url_df

if __name__ == '__main__':
    main()
