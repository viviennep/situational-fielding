import numpy as np, polars as pl, requests, json, pathlib, pickle as pkl, duckdb
from catboost import CatBoostClassifier
from data_scripts.statcast import get_statcast, add_playids
from data_scripts.outfield import retrieve_of_plays
from data_scripts.player_info import get_player_bios
from data_scripts.wall import calc_wall_properties
from expected_win_probability import row_wise_xwp
cl = pl.col
data_dir = pathlib.Path('data')
table_dir = pathlib.Path('tables')
model_dir = pathlib.Path('models')

# base state to int mapper
bit_mapper = {'---': 0b000,
              '1--': 0b001,
              '-1-': 0b010,
              '--1': 0b100,
              '11-': 0b011,
              '1-1': 0b101,
              '-11': 0b110,
              '111': 0b111}

# column lists for prettier polars
game_state   = ['inn_ind','half_ind','base_cd','outs_when_up','run_diff']
positions    = ['pitcher', 'catcher', 
               'first base', 'second base', 'third base', 'shortstop',
               'left(?:-| )field', 'center(?:-| )field', 'right(?:-| )field']
outs         = ['double_play','sac_bunt_dboule_play','fielders_choice',
                'grounded_into_double_play','force_out','sac_bunt',
                'fielders_choice_out','field_out','sac_fly_double_play','triple_play','sac_fly']
fielder_cols = [f'fielder_{i}' for i in range(2,10)]

sc_df     = get_statcast() # load statcast data
df        = add_playids(sc_df) # add play_ids to it
of_plays  = retrieve_of_plays(df) # grab outfield plays for all outfielders in the sc data
fences_df = pl.read_parquet(f'{data_dir}/fences-lidar.parquet') # get fence lidar measurements
wp_df     = pl.scan_parquet(f'{table_dir}/p(win|inn,half,base,out,rdiff).parquet') # get win prob table
li_df     = pl.scan_parquet(f'{table_dir}/leverage-index.parquet') # get leverage index table

# make the df lazy for some speedups
df = df.lazy()

# grab all player ids in the sc_df and load their bio info (just need names)
playerids = np.unique(df.select(*(cl(f'fielder_{i}') for i in range(2,10)),'batter','pitcher').collect().to_numpy())
player_df = get_player_bios(playerids)

# merge the outfield plays into the statcast df
df = df.join(of_plays.lazy(),on='play_id',how='left')

# get all the wall info for outfield plays
wall_prop_dict = calc_wall_properties(df.filter(cl('start_pos_x').is_not_null()).collect(),fences_df)
df = df.join(pl.LazyFrame(wall_prop_dict),on='play_id',how='left')

# mangle the df into how I want it for wpa/wpoe calcs!
# i don't want to comment any of this but I think each bit is self explanatory
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

# load in the models
with open(model_dir / 'outcome-given-hit.pkl','rb') as f: 
    outcome_given_hit = pkl.load(f)

# OF catch prob model
with open(model_dir / 'catch-prob.pkl','rb') as f: 
    catch_prob = pkl.load(f)

# IF out prob model
with open(model_dir / 'out-prob.pkl','rb') as f: 
    out_prob = pkl.load(f)

# Baserunner advancement model
base_adv = CatBoostClassifier().load_model(model_dir / 'baserunner-advancement.cbm')

# win prob table
wp_table = np.load(table_dir / 'p(win|inn,half,base,out,rdiff).npy')

# get p(bip outcome | theta, ev, la, hit)
X = df.select('home_team','theta','launch_speed','launch_angle').collect().to_numpy()
pred_outcome = outcome_given_hit.predict_proba(X)

# do OF plays first
of_play = df.filter(cl('is_of_play'))
of_features = ['dist','angle','hang_time','wall_dist_start',
               'wall_dist_land','wall_dist_ball_dir','wall_min_dist','wall_height']
X = of_play.select(of_features).collect().to_numpy()
of_play = of_play.with_columns(out_prob = catch_prob.predict_proba(X)[:,-1])
df = df.join(of_play.select('play_id','out_prob'),on='play_id',how='left')

# then do IF plays, IF plays are extremely fake right now the model sucks & they shouldnt be used
if_features = ['if_fielding_alignment','stand','theta','launch_speed','launch_angle']
X = df.select(if_features).collect().to_numpy()
if_out_prob = out_prob.predict_proba(X)[:,-1]

# add in IF out probs & bip outcome probs
df = df.with_columns(out_prob=pl.when(cl('out_prob').is_not_null())
                                .then('out_prob')
                                .otherwise(if_out_prob),
                     **dict(zip(['p_1b','p_2b','p_3b','p_hr'],pred_outcome.T)))

xwp_cols = ('inn_ind','half_ind','outs_when_up','base_cd','run_diff',
            'theta','launch_speed','launch_angle',
            'out_prob','p_1b','p_2b','p_3b','p_hr')

curry = lambda r: row_wise_xwp(r,base_adv,wp_table)
df = df.with_columns(xwp = pl.struct(xwp_cols).map_elements(curry,return_dtype=pl.Float64))
df = (df.with_columns(wpa_dir = 1-2*cl('half_ind'),
                      visra = cl('is_out')-cl('out_prob'),
                      scsra = cl('is_out')-cl('catch_rate'))
        .with_columns(wpa = (cl('next_wp')-cl('wp'))*cl('wpa_dir'),
                      xwpa = (cl('xwp')-cl('wp'))*cl('wpa_dir'),
                      wpoe = (cl('next_wp')-cl('xwp'))*cl('wpa_dir'))
        .with_columns(wpali = pl.when(cl('li').eq(0)).then(0).otherwise(cl('wpa')/cl('li')),
                      wpoeli = pl.when(cl('li').eq(0)).then(0).otherwise(cl('wpoe')/cl('li'))))

df = df.collect()

# hive partitioned data
df.write_parquet(f"{data_dir}/daily_data/",
                 use_pyarrow=True,
                 pyarrow_options={'partition_cols':['game_year','game_date'],
                                  'existing_data_behavior':'delete_matching'})

con = duckdb.connect(data_dir / 'leaderboard.duckdb')
con.execute(f"""
    create or replace view all_plays as
    select *
    from read_parquet('data/daily_data/*/*/*.parquet',
                      hive_partitioning=True);
""")

con.execute("""
    create or replace table leaderboard as
    with main_stats as (
        select 
            game_year as season,
            resp_fielder_name as fielder_name,
            count(*)          as plays,
            sum(visra)        as vioaa,
            sum(scsra)        as scoaa,
            sum(wpoe)         as wpoe,
            sum(wpoeli)       as wpoeli,
            min(
             date_diff('year',cast(resp_fielder_bday as date),make_date(game_year,7,1))
            ) as age,
            case
             when count(distinct fld_team)=1 then min(fld_team)
             else cast(count(distinct fld_team) as varchar) || 'TM'
            end as team
        from all_plays
        where is_of_play
        group by game_year,resp_fielder_name),
    pos_counts as (
        select
            game_year          as season,
            resp_fielder_name  as fielder_name,
            resp_fielder       as pos_code,
            count(*)           as cnt
        from all_plays
        where is_of_play
        group by game_year, resp_fielder_name, resp_fielder),
    primary_pos as (
        select
            season,
            fielder_name,
            pos_code as primary_pos_code
        from (
            select
                season,
                fielder_name,
                pos_code,
                cnt,
                row_number()
                    over(
                        partition by season, fielder_name
                        order by cnt desc, pos_code
                    ) as rn
            from pos_counts
        )
        where rn=1
    )
    select
        main_stats.*,
        primary_pos.primary_pos_code,
        case primary_pos.primary_pos_code
            when 2 then 'C'
            when 3 then '1B'
            when 4 then '2B'
            when 5 then '3B'
            when 6 then 'SS'
            when 7 then 'LF'
            when 8 then 'CF'
            when 9 then 'RF'
            else 'XX'
        end as primary_position
    from main_stats left join primary_pos using (season,fielder_name);
""")

con.close()

