import numpy as np, polars as pl, requests, json
cl = pl.col

''' Merging Retrosheet (chadwick) & Statcast data
    Retrosheet: 1 row per 'event' (PA, SB, CS, pickoff, etc.)
    Statcast: 1 row per pitch
    Obviously, no shared unique identifier columns, so we need to build them.
    Retrosheet has:
      game_id: str in format TTTYYYYMMDDG,
               TTT home team abbr, 
               YYYYMMDD game date, 
               G game number (1 or 2 for doubleheaders, 0 for normal)
      bothteams_pa_ct: int count of PA in games
      bat_event_fl: flag to filter out batter events from basepath events
    Statcast has:
      game_pk: unique game identifier
      at_bat_number: the number of the at bat (PA) in the game
    Add in schedule information from statsapi to statcast df on game_pk
    (to get double header info) and you can reconstruct retro game_id
    Merge on retro game_id & game pa count
'''

# Retrosheet data
rdf = pl.scan_parquet('../../data/2021-2024-full-retrosheet.parquet')
rdf = rdf.rename({i: i.lower() for i in rdf.collect_schema().names()})

# Statcast data
sdf = pl.read_parquet('../../data/2021-2024-sc-with-playid.parquet')

# Get schedule needed for joining data
years = sdf.select('game_year').unique().to_numpy().squeeze()
combined_schedules = []
for year in years:
    start_date    = sdf.filter(cl('game_year').eq(year)).select('game_date').min().item().strftime('%Y-%m-%d')
    end_date      = sdf.filter(cl('game_year').eq(year)).select('game_date').max().item().strftime('%Y-%m-%d')
    schedule_url  = "https://statsapi.mlb.com/api/v1/schedule?hydrate=&sportId=1&"
    schedule_url += f"startDate={start_date}&endDate={end_date}"
    res           = requests.get(schedule_url,headers={'User-Agent':'Mozilla/5.0'})
    schedule      = json.loads(res.content.decode('utf-8'))['dates']
    combined_schedules += [k for i in schedule for j in i['date'] for k in i['games']]

# Extract what we need (game number of day, doubleheader flag) from this
schedule = (pl.DataFrame(combined_schedules)
              .with_columns(gameDate=cl('gameDate').str.to_datetime('%Y-%m-%dT%H:%M:%SZ'),
                            retro_game_number=pl.when(cl('doubleHeader').eq('N'))
                                                .then(0)
                                                .otherwise('gameNumber'))
              .select(cl('gamePk').alias('game_pk'),
                      'retro_game_number',
                      'gameDate',
                      'reverseHomeAwayStatus',
                      'doubleHeader')
              .unique()
              .sort('gameDate')
              .group_by('game_pk',maintain_order=True)
              .agg(pl.last('retro_game_number'),pl.last('gameDate'),pl.last('doubleHeader'),
                   pl.last('reverseHomeAwayStatus')))

# Team abbr. are different between the two sources
statcast_to_retro = {'ATL': 'ATL', 'AZ':  'ARI', 'BAL': 'BAL',
				     'BOS': 'BOS', 'CHC': 'CHN', 'CIN': 'CIN',
				     'CLE': 'CLE', 'COL': 'COL', 'CWS': 'CHA',
				     'DET': 'DET', 'HOU': 'HOU', 'KC':  'KCA',
				     'LAA': 'ANA', 'LAD': 'LAN', 'MIA': 'MIA',
				     'MIL': 'MIL', 'MIN': 'MIN', 'NYM': 'NYN',
				     'NYY': 'NYA', 'OAK': 'OAK', 'ATH': 'OAK',
				     'PHI': 'PHI', 'PIT': 'PIT', 'SD':  'SDN',
				     'SEA': 'SEA', 'SF':  'SFN', 'STL': 'SLN',
				     'TB':  'TBA', 'TEX': 'TEX', 'TOR': 'TOR',
				     'WSH': 'WAS'}

# Create the join fields
sdf = (sdf.filter(cl('events').is_not_null(),cl('game_type').eq('R'))
          .join(schedule,on='game_pk')
          .with_columns(game_date_string = cl('game_date').dt.strftime('%Y%m%d'),
					    home_team_retro = pl.when('reverseHomeAwayStatus')
                                            .then('away_team')
                                            .otherwise('home_team')
                                            .replace_strict(statcast_to_retro),
                        inn = cl('inning')+pl.when(cl('inning_topbot').eq('Bot')).then(0.5).otherwise(0))
		  .with_columns(game_id_no_gn = cl('home_team_retro')+cl('game_date_string'))
          .with_columns(actual_doubleheader = cl('retro_game_number').eq(2).any().over('game_id_no_gn'))
          .with_columns(game_id = cl('game_id_no_gn') + pl.when('actual_doubleheader')
                                                          .then('retro_game_number')
                                                          .otherwise(0).cast(str))
          .sort('gameDate','game_pk','inn','outs_when_up','at_bat_number','pitch_number'))

# Join statcast to retrosheet
# game_id mapping works completely 1-to-1 for me!
# however, certain plate appearances are missing:
#   - all intentional walks, they don't exist in the savant csv source
#   - some random missing pitches with no explanation
#     for example: https://baseballsavant.mlb.com/sporty-videos?playId=1b272191-8a83-40cd-af6e-3ebf7c92387c
#     whatever happens after this (a ball 4) is missing
mdf = (rdf.filter(cl('bat_event_fl').eq('T'),cl('game_type').eq('R'))
          .join(sdf.lazy(), left_on=['game_id','bothteams_game_pa_ct'],
                            right_on=['game_id','at_bat_number'],
                            how='left'))
mdf.collect().write_parquet('../../data/2021-2024-retro-and-statcast.parquet')



