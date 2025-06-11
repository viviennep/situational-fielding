import numpy as np, polars as pl, requests, pathlib
from wall import calc_wall_properties
from player_info import get_player_bios
cl = pl.col
data_dir = pathlib.Path(__file__).resolve().parent.parent / 'data'
table_dir = pathlib.Path(__file__).resolve().parent.parent / 'tables'
model_dir = pathlib.Path(__file__).resolve().parent.parent / 'models'

bit_mapper = {'---': 0b000,
              '1--': 0b001,
              '-1-': 0b010,
              '--1': 0b100,
              '11-': 0b011,
              '1-1': 0b101,
              '-11': 0b110,
              '111': 0b111}

df        = pl.scan_parquet(data_dir / '2021-2024-sc-with-playid.parquet')
of_plays  = pl.scan_parquet(data_dir / '*-of-plays.parquet')
fences_df = pl.read_parquet(data_dir / 'fences-lidar.parquet')
wp_df     = pl.scan_parquet(table_dir / 'p(win|inn,half,base,out,rdiff).parquet')
li_df     = pl.scan_parquet(table_dir / 'leverage-index.parquet')

playerids = np.unique(df.select(*(cl(f'fielder_{i}') for i in range(2,10)),'batter','pitcher').collect().to_numpy())
player_df = get_player_bios(playerids)

df = df.join(of_plays,on='play_id',how='left')

wall_prop_dict = calc_wall_properties(df.filter(cl('start_pos_x').is_not_null()).collect(),fences_df)

df = df.join(pl.LazyFrame(wall_prop_dict),on='play_id',how='left')

game_state = ['inn_ind','half_ind','base_cd','outs_when_up','run_diff'] # for wp/li

positions = ['pitcher', 'catcher', 
            'first base', 'second base', 'third base', 'shortstop',
            'left(?:-| )field', 'center(?:-| )field', 'right(?:-| )field']

outs = ['double_play','sac_bunt_dboule_play','fielders_choice',
        'grounded_into_double_play','force_out','sac_bunt',
        'fielders_choice_out','field_out','sac_fly_double_play','triple_play','sac_fly']

fielder_cols = [f'fielder_{i}' for i in range(2,10)]

df = (df.with_columns(game_date=cl('game_date').dt.date(),
                      fielder_name=cl('name_display_first_last'),
                      resp_fielder = pl.when(cl('pos').is_null())
                                       .then(pl.concat_list(cl('des').str.find(p) for p in positions).list.arg_min())
                                       .otherwise('pos'),
                      is_out = cl('events').is_in(outs),
                      is_of_play = cl('start_pos_x').is_not_null(),
                      inn_ind = (cl('inning')-1).clip(0,9),
                      half_ind = (1-cl('inning_topbot').eq('Top')).cast(pl.Int64),
                      run_diff = cl('home_score')-cl('away_score'),
                      hc_x_ft = 2.495671*( cl('hc_x')-125.42), 
                      hc_y_ft = 2.495671*(-cl('hc_y')+198.27), 
                      base_state = pl.when(cl('on_1b').is_not_null()).then(pl.lit('1')).otherwise(pl.lit('-')) +
                                   pl.when(cl('on_2b').is_not_null()).then(pl.lit('1')).otherwise(pl.lit('-')) +
                                   pl.when(cl('on_3b').is_not_null()).then(pl.lit('1')).otherwise(pl.lit('-')),
                      if_fielding_alignment=pl.when(cl('if_fielding_alignment').is_not_null())
                                              .then('if_fielding_alignment')
                                              .otherwise(pl.lit('Unknown')))
        .with_columns(base_cd = cl('base_state').replace_strict(bit_mapper),
                      fld_team = pl.when(cl('half_ind').eq(1)).then('away_team').otherwise('home_team'),
                      theta = pl.arctan2('hc_x_ft','hc_y_ft'),
                      hc_dist = (cl('hc_x_ft')**2+cl('hc_y_ft')**2)**0.5)
        .sort('game_date','at_bat_number','pitch_number')
        .with_columns(next_inn_ind = cl('inn_ind').shift(-1).over('game_pk'),
                      next_half_ind = cl('half_ind').shift(-1).over('game_pk'),
                      next_base_cd = cl('base_cd').shift(-1).over('game_pk'),
                      next_outs_when_up = cl('outs_when_up').shift(-1).over('game_pk'),
                      next_run_diff = cl('run_diff').shift(-1).over('game_pk'),
                      backup_resp_fielder = pl.when((cl('hc_dist')>250) & (cl('theta') < -np.pi/6)).then(7)
                                              .when((cl('hc_dist')>250) & cl('theta').is_between(-np.pi/6,np.pi/6)).then(8)
                                              .when((cl('hc_dist')>250) & (cl('theta') > np.pi/6)).then(9)
                                              .when(cl('hc_dist').is_between(10,250) & (cl('theta') < -np.pi/8)).then(5)
                                              .when(cl('hc_dist').is_between(10,250) & cl('theta').is_between(-np.pi/8,0)).then(6)
                                              .when(cl('hc_dist').is_between(10,250) & cl('theta').is_between(0,np.pi/8)).then(5)
                                              .when(cl('hc_dist').is_between(10,250) & (cl('theta') > np.pi/8)).then(3)
                                              .otherwise(2))
        .with_columns(resp_fielder = pl.when(cl('resp_fielder').is_not_null())
                                       .then('resp_fielder')
                                       .otherwise('backup_resp_fielder'))
        .with_columns(resp_fielder_id = pl.concat_list(f'fielder_{i}' for i in range(2,10)).list.get(cl('resp_fielder')-2))
        .filter(cl('events').is_not_null(),cl('game_type').eq('R'))
        .filter(cl('theta').is_not_null(),cl('launch_speed').is_not_null(),cl('launch_angle').is_not_null())
        .join(player_df.lazy().select(cl('id').alias('resp_fielder_id'),
                                      cl('fullName').alias('resp_fielder_name'),
                                      cl('birthDate').alias('resp_fielder_bday')),
              on='resp_fielder_id')
        .join(wp_df,on=game_state,how='left')
        .join(li_df,on=game_state,how='left')
        .join(wp_df.rename({'wp':'next_wp'}),left_on=[f"next_{i}" for i in game_state],right_on=game_state,how='left')
        .with_columns(next_wp = pl.when(cl('next_wp').is_not_null()).then('next_wp')
                                  .when((cl('post_home_score')-cl('post_away_score'))>0).then(pl.lit(1.))
                                  .when((cl('post_home_score')-cl('post_away_score'))<0).then(pl.lit(0.))
                                  .otherwise(cl('wp').round())) # final otherwise only triggers in 1 instance: walk off balk
        .select('play_id','game_date','game_year','home_team','away_team','fld_team','game_pk',
                'inning','inning_topbot','outs_when_up','base_state','run_diff','balls','strikes',
                'inn_ind','half_ind','base_cd','wp','li','next_wp','is_of_play','is_out',
                'next_inn_ind','next_half_ind','next_base_cd','next_outs_when_up','next_run_diff',
                cl('player_name').alias('pitcher_name'),'fielder_name','batter','pitcher','stand','events','des',
                'theta','launch_speed','launch_angle','hc_x','hc_y','hc_x_ft','hc_y_ft','hc_dist',
                'start_pos_x','start_pos_y','landing_pos_x','landing_pos_y','hang_time',
                'fielder_2','fielder_3','fielder_4','fielder_5','fielder_6',
                'fielder_7','fielder_8','fielder_9','if_fielding_alignment',
                'resp_fielder','resp_fielder_id','resp_fielder_name','resp_fielder_bday',
                'post_home_score','post_away_score','post_bat_score','post_fld_score',
                'catch_rate', 'angle', 'dist', 'wall_dist_start', 'wall_dist_land', 
                'wall_dist_ball_dir', 'wall_min_dist', 'wall_height'))

df = df.collect()

df.write_parquet(data_dir / 'test-data-for-xwp-model.parquet')

#for date, sub in lf.collect().group_by("game_date"):
#    season = date.year
#    path   = f"../data/play_dates/season={season}/{date}.parquet"
#    sub.write_parquet(path, compression='zstd')

#f,ax = plt.subplots(subplot_kw={'projection':'polar'})
#ax.scatter(*df.filter(cl('resp_fielder').is_null()).select('theta',(cl('hc_x_ft')**2+cl('hc_y_ft')**2)**0.5).to_numpy().T)
#ax.set_theta_zero_location('N')
#ax.set_aspect('equal')
#plt.show()




