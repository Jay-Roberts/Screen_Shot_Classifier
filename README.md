# Screen Shot Classifier
These codes are modifications of the master to better be used in a cloud computing setting. 
## Top100Games.py:
Creates 'Top100Games.csv', a csv of the current top 100 games on Steam. The csv has columns 'GAME', 'STEAM ID', 'IMAGES'. 

The games SDK Source SDK Base 2013 Multiplayer and Wall paper manager have been removed since the first does not have a community page and the second is not a game.


## CommunityImages.py:
For each game in 'Top100Games.csv' it collects the urls of game images from the game's community page. The csv has columns 'URL' and 'DOWNLOADED'. Parallelized by _gameID_.

## Downloader.py:
Downloads images and saves locally from urls in _ID_ _.csv. The csv must have columns 'URL' and 'DOWNLOADED'. 
Assumes the csvs are in _sourcedir_/_ID_/_ID__urls.csv. Parallelized by _ID_.

## Processor.py:
Processes image files to be used with tensorflow. Resizes images and splits them into (train,test,validate) files. Can save locally or to a Google-Cloud-Bucket. Images are processed and saved/uploaded in chunks. Parallelized by chunk per _ID_.

Assumes images are in _sourcedir_/_ID_/


## requirements.txt:
Required python packages.
