from lxml import html
import requests
import pandas as pd



# Makes a CSV of the top 100 games on steam with their steam ids
# DataFrame is of the form {'GAME': Games,'STEAM ID': GamesID}
# Currently adds an extra index column for no reason

def main():
    top_games = requests.get('http://store.steampowered.com/stats/')

    # html.fromstring expects bytes as input so we use content
    # instead of text.

    top_tree= html.fromstring(top_games.content)

    Games = top_tree.xpath('//a[@class="gameLink"]/text()')
    GamesID = top_tree.xpath('//a[@class="gameLink"]/@href')


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
        


    Game_list= pd.DataFrame({'GAME': Games,'STEAM ID': GamesID})

    Game_list['STEAM ID'] = list(map(get_appID,Game_list['STEAM ID']))

    Game_list.to_csv('Top100Games.csv', index= False)

if __name__ == '__main__':
    main()