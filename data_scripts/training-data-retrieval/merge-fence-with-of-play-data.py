import numpy as np, polars as pl, os, sys
cl = pl.col

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
sys.path.insert(0, parent_dir)
from wall import calc_dist2wall, calc_wall_properties

df = pl.scan_parquet('../../data/2021-2024-sc-with-playid.parquet')
fences_df = pl.read_parquet('../../data/fences-lidar.parquet')
of_plays = pl.scan_parquet([f"../../data/{i}-of-plays.parquet" for i in range(2021,2025)]) # 2021-2024

of_plays = of_plays.join(df,on=('game_pk','play_id')).collect()

wall_prop_dict = calc_wall_properties(of_plays,fences_df,verbose=True)

of_plays = of_plays.with_columns(
    angle=wall_prop_dict['angle'],
    dist=wall_prop_dict['dist'],
    hit_dist=wall_prop_dict['hit_dist'],
    wall_dist_hit=wall_prop_dict['wall_dist_hit'],
    wall_height_hit=wall_prop_dict['wall_height_hit'],
    wall_dist_start=wall_prop_dict['wall_dist_start'],
    wall_dist_land=wall_prop_dict['wall_dist_land'],
    wall_dist_ball_dir=wall_prop_dict['wall_dist_ball_dir'],
    wall_min_dist=wall_prop_dict['wall_min_dist'],
    wall_height=wall_prop_dict['wall_height']
)

of_plays.write_parquet('../../data/2021-2024-of-plays-with-wall.parquet')

