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
- In 'Gameurls/game_name' a text file containing the URLs of images from the games steam community is created/updated.
 """

from lxml import html
from lxml import etree
import requests
import os
import Top100Games_cld
import pandas as pd
def main(Num_scrolls = 5, save = True):

    Num_scrolls_str = str(Num_scrolls)
    # Get the top Num_games info 

    if not os.path.exists('Top100Games_cld.csv'):
        print('Creating Top 100 List...')
        Game_df = Top100Games_cld.main()
        Game_df.to_csv('Top100Games_cld.csv', index = False)
    else:
        Game_df = pd.read_csv('Top100Games_cld.csv')
    if not os.path.isdir('Gameurls'):
        os.makedirs('Gameurls')
    
    
    # Make a dictionary to hold all the game_url dataframes
    # keys will be appID as a string
    Game_urls = {}

    for index,row in Game_df.iterrows():
        appID, name = int(row['STEAM ID']), row['GAME']
        game = str(appID)

        # Make the an entry in the dicitionary for the game urls
        Game_urls[str(appID)] = []

        print('Game: ', name)
        for scroll in range(Num_scrolls):
            
            # Make the game community url
            game_com = 'http://steamcommunity.com/app/'+ game +'/homecontent/?screenshotspage='+str(scroll)+'&numperpage=100&browsefilter=mostrecent&browsefilter=mostrecent&l=english&appHubSubSection=2&filterLanguage=default&searchText=&forceanon=1'
            
            # NOT ALL PAGES HAVE A COMMUNITY
            # See if the page exists:
            try:
                # Request the page
                game_pg = requests.get(game_com)
                print(name+' community found.')

                # Make the tree
                game_tree = html.fromstring(game_pg.content)

                #Scrape for the sources of the images displayed
                game_img_urls = game_tree.xpath('//img[@class = "apphub_CardContentPreviewImage"]/@src')

            # IF the community does not exist
            except:
                print(name + ' community not found')
                game_img_urls = ['Community Not Found']

            Game_urls[str(appID)] += game_img_urls
        
        print('Saving urls')
        # Make a directory for each game to hold a text file of the urls
        # ignore repeated urls
        game_urls = list(set(Game_urls[game]))

        #The saving path
        game_url_path = 'Gameurls/'+game
        #Check if the game has a directory
        if not os.path.isdir(game_url_path):
            os.makedirs(game_url_path)
        
        #Save as txt file that's easy to open            
        with open(game_url_path+'/'+game+'urls.txt', 'w') as f:
            for url in game_urls:
                f.write(url + '\n')
if __name__ == '__main__':
    main()
