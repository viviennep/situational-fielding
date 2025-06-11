import pybaseball as pb, polars as pl, requests, json
from pybaseball import cache
from time import time
cl = pl.col
cache.enable()

# Do this from 2021-2024 first (for training data)
# Then do just 2025
start_year = 2025
end_year = 2025

headers = {'User-Agent': 'Mozilla/5.0'}

seasons_url = "https://statsapi.mlb.com/api/v1/seasons/all?sportId=1"
res = requests.get(seasons_url,headers=headers)
seasons = pl.DataFrame(json.loads(res.content.decode('utf-8'))['seasons']).cast({'seasonId':int})

start_date = seasons.filter(cl('seasonId').eq(start_year)).select('regularSeasonStartDate').item()
end_date   = seasons.filter(cl('seasonId').eq(end_year)).select('regularSeasonEndDate').item()

# this will take a few minutes, like ~5 for me (2021-2024). takes like ~10GB too 
sc_data = pb.statcast(start_date,end_date)
sc_data = pl.from_pandas(sc_data)
sc_data = (sc_data.sort('game_pk','at_bat_number','pitch_number')
                  .with_columns(pa_pitch_number=cl('at_bat_number').cum_count().over('game_pk','at_bat_number')))
sc_data.write_parquet(f'../../data/{start_year}-{end_year}-sc.parquet')

# this will really take a while. think ~2 hours (2021-2024)
rows = []
game_pks = sc_data.select('game_pk').unique().to_numpy().squeeze()
t0 = time()
for i,gpk in enumerate(game_pks):
    print(f"{i+1:5} of {len(game_pks)} : time: {time()-t0:0.1f}")
    sa_url  = lambda x: f"https://statsapi.mlb.com/api/v1/game/{x}/playByPlay"
    res = requests.get(sa_url(gpk),headers=headers)
    data = json.loads(res.content)
    for play in data['allPlays']:
        atBatIndex = play['atBatIndex']+1
        for event in play['playEvents']:
            if event['isPitch']:
                playId = event['playId']
                pa_pitch_num = event['pitchNumber']
                rows.append([gpk,atBatIndex,playId,pa_pitch_num])
    if (i+1)%100:
        playid_df = pl.DataFrame(rows,schema={'game_pk':int,'at_bat_number':int,'play_id':str,'pa_pitch_number':int})
        playid_df.write_parquet(f'../../data/sa_cache/{start_year}-{end_year}-sa_cache')
    if (i+1)%101: # second one is so if this gets interrupted the entire parquet file doesn't get lost :/ 
        playid_df = pl.DataFrame(rows,schema={'game_pk':int,'at_bat_number':int,'play_id':str,'pa_pitch_number':int})
        playid_df.write_parquet(f'../../data/sa_cache/{start_year}-{end_year}-sa_cache')

playid_df = pl.DataFrame(rows,schema={'game_pk':int,'at_bat_number':int,'play_id':str,'pa_pitch_number':int})
playid_df.write_parquet(f'../../data/{start_year}-{end_year}-playid.parquet')

# this is 100% accurate for me :)
sc_data = sc_data.join(playid_df,on=['game_pk','at_bat_number','pa_pitch_number'])
sc_data.write_parquet(f'../../data/{start_year}-{end_year}-sc-with-playid.parquet')


