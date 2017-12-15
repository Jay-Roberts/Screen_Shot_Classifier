from lxml import html
import requests
import pandas as pd

"""If save = False Returns a DataFrame of the current top 100 games on steam with their steam ids
DataFrame is of the form {'GAME': Games,'STEAM ID': GamesID}. 
Otherwise Dataframe is saved locally as Top100games_cld.csv"""    

# Go through steam top 100 games and make a list of them together with their steam AppID
# Returns a Dataframe

def main(save = True):
    print('Downloading Top 100 List')
    top_games = requests.get('http://store.steampowered.com/stats/')

    # html.fromstring expects bytes as input so we use content
    # instead of text.
    top_tree= html.fromstring(top_games.content)

    Games = top_tree.xpath('//a[@class="gameLink"]/text()')
    GamesID = top_tree.xpath('//a[@class="gameLink"]/@href')

    # Get rid of games without community pages
    # GamesID.remove('243750') # SDK Source SDK Base 2013 Multiplayer
    # GamesID.remove('431960') # 
    
    # fix the names
    Games = list(map(fix_name,Games))
    GamesID = list(map(get_appID,GamesID))
    Game_list= pd.DataFrame({'GAME': Games,'STEAM ID': GamesID, 'IMAGES': [0]*len(Games)})
    
    # Problem "games"
    # 243750 # SDK Source SDK Base 2013 Multiplayer
    # 431960 # Wall paper manager
     
    problem_ix = [GamesID.index('243750'),GamesID.index('431960')]
    Game_list  = Game_list.drop(problem_ix)
    
    if save:
        Game_list.to_csv('Top100Games_cld.csv', encoding = 'utf-8')
    else:
        return Game_list


#
# Utility functions
#
# Fix the names
def fix_name(word):

    word.encode('utf-8', 'ignore')
    return word

#These get the steamID from the trees
def get_url_terms(url):
    # skip http://
    newrl = list(url)[7:]
    terms = []
    word = ''
    
    for let in newrl:
        if let == '/':
            terms.append(word)
            word = ''
        elif newrl.index(let) == len(newrl)-1:
            word = word+let
            terms.append(word)
            word = ''
        else:
            word = word + let
    return terms

#Get the app ID from the url
def get_appID(url):
    terms = get_url_terms(url)
    spot = terms.index('app')
        
    if len(terms) > spot+1:
        return terms[spot+1]


if __name__ == '__main__':
    main()