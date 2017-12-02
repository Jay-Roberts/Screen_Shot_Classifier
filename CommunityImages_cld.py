#---------------------------------------------------------------------------------------------------------------------
#
#       GENERATE LIST OF IMAGE URL FOR TOP100GAMES
#       RETURNS DIRECTORY WITH FOLDERS FOR EACH GAME CONTAING A CSV OF IMAGE URLS AND A FOLDER OF DOWNLOADED IMAGES
#
#---------------------------------------------------------------------------------------------------------------------

"""
- Should be run with os argumetns: Num_games Num_scrolls Num_down. 
Otherwise provide these integers as a list.
- If Top100Games.csv does not exist Top100Games.py is called to create it.
- If they don't exist '/rawimages' is created and 'rawimages/game_name' directories are created 
for each game. 
- In 'rawimages/game_folder' a csv containing the URLs of images from the games steam community is created/updated.
- In 'rawimages/game_folder' '/downloads' is created.
- If the games steam community page is found Num_scrolls*Num_down images are saved in the downloads folder.abs
If not then the game's URL dataframe is updated with COMMUNITY NOT FOUND and the program continues 
 """



import os
from lxml import html
from lxml import etree
import requests
import pandas as pd
import Top100Games_cld
import sys

def main(Num_games = 5,Num_scrolls = 1,Num_down = 5):
    
    print('Games, Scrolls, Down')
    print(Num_games,Num_scrolls,Num_down)
    # Get the top Num_games info 

    if not os.path.isfile('Top100Games.csv'):
        Game_df = Top100Games_cld.main()
    
    # Make a folder to hold all the game images.
    # Check if the folder is already there
    if not os.path.exists('rawimages'):
        os.makedirs('rawimages')

    if not os.path.exists('Top100Games_cld.csv'):
        print('Creating Top 100 List...')
        Top100Games_cld.main().to_csv('Top100Games_cld.csv')

    # Make a dictionary to hold all the game_url dataframes
    Game_urls = {}
    for index,row in Game_df.iterrows():
        appID = int(row['STEAM ID'])
        name = row['GAME']

        Game_urls[str(appID)] = []
        # Fixing the name. Should change this to  ASCI package
        name = name.replace(' ','_')
        name = name.replace(':','-c-')

        print('Game: ', name)

        appID_str = str(appID)
        Num_scrolls_str = str(Num_scrolls)
        
        # Initialize the game_url Dataframe
        
        for scroll in range(Num_scrolls):
            
            # Make the games url
            g_com_url = 'http://steamcommunity.com/app/'+ appID_str +'/homecontent/?screenshotspage='+str(scroll)+'&numperpage=100&browsefilter=mostrecent&browsefilter=mostrecent&l=english&appHubSubSection=2&filterLanguage=default&searchText=&forceanon=1'
            
            # NOT ALL PAGES HAVE A COMMUNITY
            # See if the page exists:
            try:
                # Request the page
                g_page = requests.get(g_com_url)

                print(name+' community found.')

                # Get the page and make the tree
                g_page = requests.get(g_com_url) 
                g_tree = html.fromstring(g_page.content)

                #Scrape for the sources of the images displayed
                g_img_urls = g_tree.xpath('//img[@class = "apphub_CardContentPreviewImage"]/@src')

            # IF the community does not exist
            except:
                print(name + ' community not found')
                G_URLS_df['IMG URL'] = 'COMMUNITY NOT FOUND'
                G_URLS_df.to_csv(file_path)

            Game_urls[str(appID)] += g_img_urls
    Game_urls = pd.DataFrame(Game_urls)
    print(Game_urls.head())
    return Game_urls
if __name__ == '__main__':
    main()
