# Screen_Shot_Classifier

# Top100Games.py: 
- Creates a list of top 100 games on Steam with their Steam AppIDs

# CommunityImages.py:
- Uses the Top100 list to download images from the game's steam community's screenshot page.
- Image urls are stored in the file 'rawimages/game_name/game_name_URLs.csv'
- Images are stored in the folder 'rawimages/game_name/downloads'
- Can choose to not download images and only update the csv to postpone downloading for a later time.

# ImagesProcessor.py:
- Uses output of CommunityImages.py to resize all images.
- Resized images are stored as a numpy bit file in'rawimages/game_name/'.

#

The whole process can be run by calling ImagesProcessor.py.
\
More details in the comments.
