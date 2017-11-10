from lxml import html
import requests
import pandas as pd



"""Makes a CSV of the current top 100 games on steam with their steam ids
DataFrame is of the form {'GAME': Games,'STEAM ID': GamesID}. Dataframe is stored
locally as Top100games.csv"""

# Fix the names
# Change this to a ASCI package
def fix_name(word):
    # Remove Spaces
    result = word.replace(' ','_')

    # Remove Colons
    result = result.replace(':','-c-')

    # Remove trade mark symbols
    result = result.replace(u"\u2122",'')

    # Remove periods

    result = result.replace('.','p_p')

    # Remove apostrophes

    result = result.replace("'",'')

    # Remove Parentheses

    result = result.replace('(','PP-P')
    result = result.replace(')','P-PP')

    # Remove ampersand

    result = result.replace('&','AmPeR')
    
    return result

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

def get_appID(url):
    terms = get_url_terms(url)
    
    spot = terms.index('app')
    #print(terms)
    
    if len(terms) > spot+1:
        return terms[spot+1]
    

    
#
# Go through steam top 100 games and make a list of them together with their steam AppID
#

def main():

    print('Downloading Top 100 List')
    top_games = requests.get('http://store.steampowered.com/stats/')

    # html.fromstring expects bytes as input so we use content
    # instead of text.

    top_tree= html.fromstring(top_games.content)

    Games = top_tree.xpath('//a[@class="gameLink"]/text()')
    GamesID = top_tree.xpath('//a[@class="gameLink"]/@href')



    Games = list(map(fix_name,Games))
    Game_list= pd.DataFrame({'GAME': Games,'STEAM ID': GamesID})

    Game_list['STEAM ID'] = list(map(get_appID,Game_list['STEAM ID']))

    Game_list.to_csv('Top100Games.csv', index= False)

if __name__ == '__main__':
    main()