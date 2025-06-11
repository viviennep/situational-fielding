import numpy as np, requests, polars as pl

def get_player_bios(ids):
    people = []
    for i in np.array_split(ids,len(ids)//300):
        api_url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str,i))}"
        res = requests.get(api_url,headers={'UserAgent':'Mozilla'}).json()
        people += res['people']
    return pl.DataFrame(people)
