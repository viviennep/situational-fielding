import numpy as np, polars as pl, pathlib, duckdb, pickle as pkl
from catboost import CatBoostClassifier
from expected_win_probability import row_wise_xwp
cl = pl.col
data_dir  = pathlib.Path('data')
table_dir  = pathlib.Path('tables')
model_dir  = pathlib.Path('models')

# load in the models
with open(model_dir / 'outcome-given-hit.pkl','rb') as f: 
    outcome_given_hit = pkl.load(f)

# Baserunner advancement model
base_adv = CatBoostClassifier().load_model(model_dir / 'baserunner-advancement.cbm')

# win prob table
wp_table = np.load(table_dir / 'p(win|inn,half,base,out,rdiff).npy')

df = pl.read_parquet('data/daily_data/',hive_partitioning=True)

X = df.select('home_team','theta','launch_speed','launch_angle').to_numpy()
pred_outcome = outcome_given_hit.predict_proba(X)

df = df.with_columns(**dict(zip(['p_1b','p_2b','p_3b','p_hr'],pred_outcome.T)))

pid = '1ec756b6-fb8b-3044-a58b-eb031f62eb50'

cols = ('inn_ind','half_ind','outs_when_up','base_cd','run_diff',
        'theta','launch_speed','launch_angle',
        'out_prob','p_1b','p_2b','p_3b','p_hr')

curry = lambda r: row_wise_xwp(r,base_adv,wp_table)
df = df.with_columns(xwp = pl.struct(cols).map_elements(curry,return_dtype=pl.Float64))
df = (df.with_columns(wpa_dir = 1-2*cl('half_ind'),
                      visra = cl('is_out')-cl('out_prob'),
                      scsra = cl('is_out')-cl('catch_rate'))
        .with_columns(wpa = (cl('next_wp')-cl('wp'))*cl('wpa_dir'),
                      xwpa = (cl('xwp')-cl('wp'))*cl('wpa_dir'),
                      wpoe = (cl('next_wp')-cl('xwp'))*cl('wpa_dir'))
        .with_columns(wpali = pl.when(cl('li').eq(0)).then(0).otherwise(cl('wpa')/cl('li')),
                      wpoeli = pl.when(cl('li').eq(0)).then(0).otherwise(cl('wpoe')/cl('li'))))

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


