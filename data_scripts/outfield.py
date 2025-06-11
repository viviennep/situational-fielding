import numpy as np, polars as pl, requests
from functools import lru_cache
cl = pl.col

def identify_outfielders(df):
    outfielders = np.unique(df.select('fielder_9','fielder_8','fielder_7').to_numpy().astype('f'))
    outfielders = outfielders[~np.isnan(outfielders)].astype(int)
    return outfielders

@lru_cache
def get_outfielder_plays(fielder,year):
    url_dict = {'playerId': f"{fielder:0.0f}",
                'season'  : year,
                'playerType': 'fielder'}
    url  = f"https://baseballsavant.mlb.com/player-services/range?" 
    url += '&'.join(f'{i}={j}' for i,j in url_dict.items())
    plays = requests.get(url, headers={'UserAgent':'Mozilla'}).json()
    return plays

def retrieve_of_plays(df):
    outfielders = identify_outfielders(df)
    year = df.select('game_year').unique()[0].item()
    all_of_plays = []
    for i,fielder in enumerate(outfielders):
        print(f"{i+1:>5} of {len(outfielders):>5}")
        plays = get_outfielder_plays(fielder,year)
        all_of_plays += plays
    of_plays = pl.DataFrame(all_of_plays).cast({'catch_rate':float,
                                                'sprint_speed':float,
                                                'hang_time':float})
    return of_plays

