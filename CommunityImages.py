#---------------------------------------------------------------------------------------------------------------------
#
#       GENERATE LIST OF IMAGE URL FOR TOP100GAMES
#       RETURNS DIRECTORY WITH FOLDERS FOR EACH GAME CONTAING A CSV OF IMAGE URLS AND A FOLDER OF DOWNLOADED IMAGES
#
#---------------------------------------------------------------------------------------------------------------------



import os
from lxml import html
from lxml import etree
import requests
import pandas as pd
import Top100Games

# If Top100Games.csv does not exist Top100Games.py is called to create it.
# "rawimages"  and "rawimages/game_name" directories are created. 
# In each game_name folder a csv containing the URLs of images from the games steam community is created/updated.
# If download = True images from this URL list are downloaded and stored in "downloads" folder 

def main():
    # Set Parameters

    Num_scrolls = 1    # Number of scrolls to call
    Num_games = 5    # Number of Games to use. Ordered by most popular)
    Num_down = 20     # Number of Games to Download
    download = True   # Decides if images are downloaded. If False only URLs are updated
    


    # Get the top Num_games info 
    Game_df = pd.read_csv('/home/jay/Desktop/Codes/WebScrapping/Top100Games.csv', nrows = Num_games)

    # Make a folder to hold all the game images.

    # Check if the folder is already there
    if not os.path.exists('rawimages'):
        os.makedirs('rawimages')

    if not os.path.exists('Top100Games.csv'):
        print('Creating Top 100 List...')
        Top100Games.main()

    for index,row in Game_df.iterrows():
        appID = int(row['STEAM ID'])
        name = row['GAME']
        name = name.replace(' ','_')
        name = name.replace(':','-c-')

        print('Game: ', name)

        appID_str = str(appID)
        Num_scrolls_str = str(Num_scrolls)
        

        

        # Make a folder for each game in the rawimages folder
        # Folder name is appID to avoid nonsense names

        fold_name = str(appID)
        image_folder = 'rawimages/'+fold_name
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        

        # Make a file to hold all the URLs So you can search later and not download repeats

        file_path = 'rawimages/'+fold_name +'/'+fold_name+'_URLs'
        if not os.path.isfile(file_path):
            # If we haven't already made a URL list make one
            G_URLS_df = pd.DataFrame({'IMG URL': []})
            G_URLS_df.to_csv(file_path, index = False)
        
        else:
            G_URLS_df = pd.read_csv(file_path)
    
        # Loop through the number of scrolls

        for scroll in range(Num_scrolls):
            
            # Make the games url
            g_url = 'http://steamcommunity.com/app/'+ appID_str +'/homecontent/?screenshotspage='+str(scroll)+'&numperpage=100&browsefilter=mostrecent&browsefilter=mostrecent&l=english&appHubSubSection=2&filterLanguage=default&searchText=&forceanon=1'
            
            #print(g_url)

            # NOT ALL PAGES HAVE A COMMUNITY
            # See if the page exists:
            

            try:
                # Request the page
                g_page = requests.get(g_url)

                print(name+' community found.')
                g_page = requests.get(g_url)
                

                g_tree = html.fromstring(g_page.content)

                #Scrape for the sources of the images displayed

                g_img_urls = g_tree.xpath('//img[@class = "apphub_CardContentPreviewImage"]/@src')

                

                new_urls_df = pd.DataFrame({'IMG URL': g_img_urls})
                

                # Add the urls to the old ones

                G_URLS_df =G_URLS_df.append(new_urls_df)

                # Make a column to check if there are duplicates

                G_URLS_df['DUPLICATE'] = G_URLS_df.duplicated()

                # Pick out only those that were not duplicates

                G_URLS_df = G_URLS_df[G_URLS_df['DUPLICATE'] == False]

                # Drop the duplicate column
                G_URLS_df = G_URLS_df.drop('DUPLICATE', axis = 1)
                
                # Write the csv
                G_URLS_df.to_csv(file_path, index = False)
                
                # Download the images from the urls
                if download:
                    limit = min([Num_down, len(g_img_urls)])

                    G_URLS_df_cut = G_URLS_df.iloc[:limit]
                    print('Downloading Images')

                    # Make a download folder inside the game folder

                    new_folder = image_folder +'/downloads/'
                    

                    if not os.path.isdir(new_folder):
                        os.makedirs(new_folder)
                    
                    img_label = len(os.listdir(new_folder))
                    for ii, G_URLS_row in G_URLS_df_cut.iterrows():
                        
                        
                        image_name = appID_str+'_'+ str(ii+img_label)

                        row_url = G_URLS_row['IMG URL']
                        img_data = requests.get(row_url).content

                        print(ii)

                        with open(new_folder+ image_name +".jpg", 'wb') as handler:
                            handler.write(img_data)

    # IF the community does not exist
            except:
                print(name + ' community not found')
                G_URLS_df['IMG URL'] = 'COMMUNITY NOT FOUND'
                G_URLS_df.to_csv(file_path+'.csv')

if __name__ == '__main__':
    main()
