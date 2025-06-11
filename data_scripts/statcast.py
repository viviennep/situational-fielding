import numpy as np, polars as pl, pybaseball as pb, duckdb, requests, time, pathlib
from functools import lru_cache
pb.cache.enable()
cl = pl.col
here = pathlib.Path(__file__).resolve().parent

def get_season_start_end_dates(start_year,end_year=None):
    if end_year is None: end_year = start_year
    seasons_url = "https://statsapi.mlb.com/api/v1/seasons/all?sportId=1"
    res = requests.get(seasons_url,headers={'UserAgent':'Mozilla'}).json()
    seasons = pl.DataFrame(res['seasons']).cast({'seasonId':type(start_year)})
    start_date = seasons.filter(cl('seasonId').eq(start_year)).select('regularSeasonStartDate').item()
    end_date   = seasons.filter(cl('seasonId').eq(end_year)).select('regularSeasonEndDate').item()
    return start_date,end_date

def get_season_gamepks(season):
    start_date = get_season_start_end_dates(season)[0]
    end_date = time.strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date}&endDate={end_date}"
    res = requests.get(url,headers={'UserAgent':'Mozilla'}).json()
    games = [g['gamePk'] for d in res['dates'] for g in d['games'] if g['status']['statusCode']=='F'
                                                                   and g['gameType']=='R']
    return np.unique(games)

def get_existing_gamepks(season):
    datadir = here.parent / 'data'
    con = duckdb.connect(datadir / 'leaderboard.duckdb')
    game_pks = con.query(f"""
        select distinct game_pk
        from all_plays
        where game_year = {season};
        """).fetchall()
    con.close()
    game_pks = [i[0] for i in game_pks]
    return game_pks

def find_required_start_date():
    season = time.strftime('%Y')
    existing_game_pks = get_existing_gamepks(season)
    all_game_pks = get_season_gamepks(season)
    game_pks = [i for i in all_game_pks if i not in existing_game_pks]
    if game_pks:
        start_date = time.strftime('%Y-%m-%d')
        s = len(game_pks)//200 if len(game_pks)>=200 else 1
        for a in np.array_split(game_pks,s):
            url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePks={','.join(map(str,a))}"
            res = requests.get(url,headers={'UserAgent':'Mozilla'}).json()
            min_start_date = min(g['officialDate'] for d in res['dates'] for g in d['games'] 
                                                   if g['status']['statusCode']=='F')
            start_date = min(start_date,min_start_date)
    else:
        start_date = get_season_start_end_dates(season)[0]
    return start_date

def get_statcast():
    start_date = find_required_start_date()
    end_date = time.strftime('%Y-%m-%d')
    season = time.strftime('%Y')
    game_pks = get_existing_gamepks(season)
    sc_data = pb.statcast(start_date,end_date)
    sc_data = pl.from_pandas(sc_data)
    # commenting this out because of how pyarrow deals with conflicting hive partitions
    #sc_data = sc_data.filter(~cl('game_pk').is_in(game_pks))
    sc_data = (sc_data.sort('game_pk','at_bat_number','pitch_number')
                      .with_columns(pa_pitch_number=cl('at_bat_number')
                                                    .cum_count()
                                                    .over('game_pk','at_bat_number')))
    return sc_data

@lru_cache
def retrieve_game_playids(gpk):
    plays = []
    sa_url  = lambda x: f"https://statsapi.mlb.com/api/v1/game/{x}/playByPlay"
    data = requests.get(sa_url(gpk),headers={'UserAgent':'Mozilla'}).json()
    for play in data['allPlays']:
        atBatIndex = play['atBatIndex']+1
        for event in play['playEvents']:
            if event['isPitch']:
                playId = event['playId']
                pa_pitch_num = event['pitchNumber']
                plays.append([gpk,atBatIndex,playId,pa_pitch_num])
    return plays

def add_playids(sc_data):
    rows = []
    game_pks = sc_data.select('game_pk').unique().to_numpy().squeeze()
    t0 = time.time()
    for i,gpk in enumerate(game_pks):
        print(f"{i+1:5} of {len(game_pks)} : time: {time.time()-t0:0.1f} {gpk}")
        plays = retrieve_game_playids(gpk)
        rows += plays
    playid_df = pl.DataFrame(rows,schema={'game_pk':int,'at_bat_number':int,'play_id':str,'pa_pitch_number':int})
    sc_data = sc_data.join(playid_df,on=['game_pk','at_bat_number','pa_pitch_number'])
    return sc_data


