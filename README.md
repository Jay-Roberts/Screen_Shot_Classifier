# Screen_Shot_Classifier
Establish a datapipeline of screenshots from the top 100 Steam games for machine learning projects.

# Setup and Example Usage

Clone the git repository into your working directory.

```git
git clone https://github.com/Jay-Roberts/Screen_Shot_Classifier/tree/Development
```

## Scraping from Steam

To scrape images from the steam website firs you must get the community pages where they are located. 

```bash
python3 Top100Games.py
```

This generates a csv with the game names and their steam gameids. From here you scrape urls from the screen-shots pages of each game.

```bash
python3 CommunityImages.py --num-games 5 \
            --num-scrolls 2 \
            --num-cores 2 \
            --save-dir Images
```

This will take the top 5 games on steam and scroll throgh their screen shots page twice, this is roughly 50 images, using 2 cores and save images in 'Images/GameID' for each game. In each of these directories a csv is created which will have columns 'URL' and 'DOWNLOADED'. The 'URL' is the url of an image and the 'DOWNLOADED' column indicates whether the image has been downloaded by the next program.

## Downloading from url lists and Processing

If you have your own urls you want to download and process make sure the csv is formatted like the output of **CommunityImages.py** and your image classes need to have a csv with class name to some unique ID formatted like **Top100Games.py** output, then you can run the following.

```bash
python3 Downloader.py --num-cores 2 \
            --chunk-size 50
```

This uses 2 cores to download images in chunks of 50. They are saved by default into GameImages/classID. These can then be processed into TFRecord files for use in deep_models.

```bash
python3 Processor.py --num-cores 2 \
                --chunk-size 50 \
                --resolution 28 28 \
                --split .6 .2 .2 

```

This uses 2 cores to process images in chunk sizes of 50 to a resolution of 28x28x3 (1 if greyscale). These images are split into train, eval, and test sets of size 60%, 20%, and 20% of database size respectively. They are saved in TFRecords/classID/{train,test,eval}_n.tfrecord. Each record has feature with a 'label' and 'image' attached.

# Docs

## Top100Games.py

Downloads the top 100 games from [Steam Top 100 List](http://store.steampowered.com/stats/). It removes the games 243750 and 431960 as they are not games. Returns a csv with columns "GAME" and "STEAMID".

## CommunityImages.py
Downloads urls of images on games' community pages for download later. For arguments descriptions run the --help flag.

Creates a csv containing the urls saved in save_dir/gameID/. The csv has columns 'URL' and 'DOWNLOADED'. 'URL' holds the image urls and 'DOWNLOADED' has values 0.0, 1.0, or 2.0, corresponding to the url having no download attempts, a successful download, or a failed download attempt respectively. 

## Downloader.py
Downloads the images from url lists. Does not need to be applied to Steam images. So long as images are stored in url_dir/classID/classID.csv with the csv having columns as in **CommunityImages.py** output this code can be used. Url data must be local but images can be downloaded to a Google Cloud Bucket. For argument descriptions run the --help flag.

Downloads the urls as raw images and places them in save_dir/classID/classID_n.jpg. Files which cannot be opened or saved as a .jpg are removed. 

## Processsor.py

Converts the raw image database into a database of uniform resolution TFRecord files split for training, evaluation, and testing.hey are saved in TFRecords/classID/{train,test,eval}_n.tfrecord. Each record has feature with a 'label' and 'image' serialized inside. Can run locally, or both pull images from and save processed images to a Google Cloud Bucket.  For argument descriptions run the --help flag.

Also creates a labes_key.csv which maps classID to model_label. This is needed in decoding the TFRecords during future tasks.

