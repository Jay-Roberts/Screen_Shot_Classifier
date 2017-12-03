# Screen Shot Classifier
These codes are modifications of the master to better be used in a cloud computing setting. 
## Top100Games_cld.py: 
Returns a DataFrame of top 100 games on Steam with their Steam AppIDs.

- INPUTS:
    {save: bool, decides whether to save the csv of games or not. _Default_ True}
- RETURNS:
    If save is False returns the DataFrame {'GAME': _games_, 'STEAM ID': _gamesID_}
    If save is True returns None but saves the above Dataframe locally as Top100Games_cld.csv


## CommunityImages_cld.py:
Uses the Top100 list to scrape image urls from the game's steam-community screenshot page. Data
is stored in DataFrame structure: {URL: the urls of images, DOWNLOADED: 0 = not downloaded 1 = downloaded}
- INPUTS:
    {Num_ scrolls: Integer for number of pages to scrape. _Default_ 6
     save: bool that decides to save the url DataFrames as a csv or not. _Default_ True}
-RETURNS:
    If save = Ture Urls are stored in the file 'Gameurls/_gameID_ / _gameID_urls.csv'. Otherwise the url DataFrame is returned.


## Downloader_cld.py:
Function to download urls from the directories created by CommunityImages_cld.py in batches.
- INPUTS:
    {resolution: tuple, the resolution of the processed image. _Default_ (224,224,3)
     first: int, index of first image to be processed (inclusive). _Default_ 0
     last: int, index of last image to be processed (not inclusive). _Default_ 100
     save: bool, if true saves the images locally. _Default_ True}
     _Note_: index is the index of the url in the _gameID__urls.csv file or DataFrame
- RETURNS:
    A list of the processed images. The images are stored as numpy arrays of size _resolution_ .

More details in the comments.
