import numpy as np, polars as pl, pickle
from sklearn.neighbors import KNeighborsClassifier
cl = pl.col

df = (pl.scan_parquet('data/2021-2024-sc-with-playid.parquet')
        .with_columns(hc_x_ft = 2.495671*(cl('hc_x')-125.42),
                      hc_y_ft = 2.495671*(-cl('hc_y')+198.27))
        .with_columns(theta = pl.arctan2('hc_x_ft','hc_y_ft'),
                      bip_outcome = cl('events').replace_strict({'single':'single', 'double':'double',
                                                                 'triple':'triple', 'home_run':'home_run'},
                                                                default='out')))

outcome_map = {'single':1,'double':2,'triple':3,'home_run':4}
Xy = (df.filter(~cl('bip_outcome').eq('out'))
        .select('theta','launch_speed','launch_angle',cl('bip_outcome').replace(outcome_map))
        .collect()
        .drop_nulls().to_numpy())
outcome_given_hit = KNeighborsClassifier(n_neighbors=25,n_jobs=-1)
outcome_given_hit.fit(Xy[:,:-1],Xy[:,-1].astype(str))

with open('models/outcome-given-hit.pkl','wb') as f:
    pickle.dump(outcome_given_hit,f)

