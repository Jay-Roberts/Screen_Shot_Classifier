import os
from lxml import html
from lxml import etree
import requests
import pandas as pd


# The URLS are of the form
# http://steamcommunity.com/app/+ "APP ID" + /homecontent/?&p= + "SCROLL NUMBER" +&numperpage= + "NUMBER PER PAGE" + &
# ENDS WITH &


Num_scrolls = 1     # Number of scrolls to call
Num_p_Page = 100     # Number of pictures per scroll page
Num_games = 5       # Number of Games to use. Ordered by most popular
download = True     # Decides if images are downloaded. If False only URLs are updated

# have the constant pieces that could be edited later if they change.

front_RL = 'http://steamcommunity.com/app/' # then put app id
#after_ID_RL = '/homecontent/?userreviewsoffset='
after_ID_RL ='/homecontent/?screenshotspage=' # then put current page number
#after_Scroll_RL = '&numperpage='
after_Num_pg = '&numperpage=100&browsefilter=mostrecent&browsefilter=mostrecent&l=english&appHubSubSection=2&filterLanguage=default&searchText=&forceanon=1'






# Get the top Num_games info 
Game_df = pd.read_csv('/home/jay/Desktop/Codes/WebScrapping/Top100Games.csv', nrows = Num_games)

# Make a folder to hold all the game images.

# Check if the folder is already there
if not os.path.exists('rawimages'):
    os.makedirs('rawimages')

for index,row in Game_df.iterrows():
    appID = int(row['STEAM ID'])
    name = row['GAME']
    name = name.replace(' ','_')

    print('Game: ', name)

    appID_str = str(appID)
    Num_scrolls_str = str(Num_scrolls)
    Num_p_Page_str = str(Num_p_Page)

    

    # Make a folder for each game in the rawimages folder

    image_folder = 'rawimages/'+name
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    

    # Make a file to hold all the URLs So you can search later and not download repeats

    url_path = 'rawimages/'+name+'/'+name+'_URLs'
    if not os.path.isfile(url_path):
        # If we haven't already made a URL list make one
        G_URLS_df = pd.DataFrame(columns=['IMG URL'])
        G_URLS_df.to_csv(url_path, index = False)
    
    else:
        G_URLS_df = pd.read_csv(url_path)
   
    # Loop through the number of scrolls

    for scroll in range(Num_scrolls):

        scroll_str = str(scroll)
        # Make the games url

        g_url = front_RL + appID_str
        g_url = g_url    + after_ID_RL     + scroll_str 
        #g_url = g_url    + after_Scroll_RL 
        #g_url = g_url    + Num_p_Page_str 
        g_url = g_url    + after_Num_pg


        # Request the page

        # NOT ALL PAGES HAVE A COMMUNITY
        # See if the page exists:
        try:
            requests.get(g_url)
            print(name+' community found.')
            g_page = requests.get(g_url)
            html.fromstring(g_page.content)

            g_tree = html.fromstring(g_page.content)

            #Scrape for the sources of the images displayed

            g_img_urls = g_tree.xpath('//img[@class = "apphub_CardContentPreviewImage"]/@src')

            new_urls_df = pd.DataFrame(columns=['IMG URL'])
            new_urls_df['IMG URL'] = g_img_urls

            # Add the urls to the old ones

            G_URLS_df =G_URLS_df.append(new_urls_df)

            # Make a column to check if there are duplicates

            G_URLS_df['DUPLICATE'] = G_URLS_df.duplicated()

            # Pick out only those that were not duplicates

            G_URLS_df = G_URLS_df[G_URLS_df['DUPLICATE'] == False]

            # Drop the duplicate column
            G_URLS_df = G_URLS_df.drop('DUPLICATE', axis = 1)
            
            # Write the csv
            G_URLS_df.to_csv(url_path, index = False)
            
            # Download the images from the urls
            if download:
                print('Downloading Images')
                for ii, G_URLS_row in G_URLS_df.iterrows():
                    
                    print(ii)
                    image_name = name + str(ii)

                    row_url = G_URLS_row['IMG URL']
                    img_data = requests.get(row_url).content
                    with open(url_path+ image_name , 'wb') as handler:
                        handler.write(img_data)

# IF the community does not exist
        except:
            print(name + ' community not found')
            G_URLS_df['IMG URL'] = 'COMMUNITY NOT FOUND'
            G_URLS_df.to_csv(url_path)


        