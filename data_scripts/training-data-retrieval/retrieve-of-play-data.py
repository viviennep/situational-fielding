import numpy as np, requests, json, polars as pl, matplotlib.pyplot as plt
from unidecode import unidecode
cl = pl.col

headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1'}

year = 2021
df = pl.read_parquet(f'../../data/2021-2024-retro-and-statcast.parquet').cast({f'fielder_{i}':pl.UInt32 for i in [7,8,9]})

outfielders = np.unique(df.filter(cl('season_id').eq(year))
                          .select('fielder_9','fielder_8','fielder_7').to_numpy().astype('f'))
outfielders = outfielders[~np.isnan(outfielders)].astype(int)

of_plays = []

for i,fielder in enumerate(outfielders):
    print(f"{i+1:>5} of {len(outfielders):>5}")
    url_dict = {'playerId': f"{fielder:0.0f}",
                'season'  : year,
                'playerType': 'fielder'}
    url  = f"https://baseballsavant.mlb.com/player-services/range?" 
    url += '&'.join(f'{i}={j}' for i,j in url_dict.items())
    json_response = requests.get(url, headers=headers)
    plays = json_response.json()
    of_plays += plays

of_plays = pl.DataFrame(of_plays).cast({'catch_rate':float,'sprint_speed':float,'hang_time':float})
of_plays.write_parquet(f"../../data/{year}-of-plays.parquet")

