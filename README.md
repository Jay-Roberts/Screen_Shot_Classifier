# Screen Shot Classifier

## Top100Games.py: 
- Returns a DataFrame of top 100 games on Steam with their Steam AppIDs
- If run with save = True saves a csv of this data

## CommunityImages.py:
- Uses the Top100 list to scrape image urls from the game's steam-community screenshot page.
- Urls are stored in the file 'Gameurls/_gameID_ / _gameID_urls.csv'
- {URL: the urls of images, DOWNLOADED: 0 = not downloaded 1 = downloaded}

## ImagesProcessor.py:
- Uses output of CommunityImages.py to resize all images.
- Creates a tensor of image data and tensor of corresponding labels.
- Saves both in 'processed' folder as images.npy and labels.npy.

More details in the comments.
