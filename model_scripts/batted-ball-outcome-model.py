import numpy as np, polars as pl, pickle, pathlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
cl = pl.col
data_dir = pathlib.Path('../data')
model_dir = pathlib.Path('../models')

#df = (pl.scan_parquet(data_dir / '2021-2024-sc-with-playid.parquet')
df = (pl.scan_parquet(data_dir / 'daily_data', hive_partitioning=True)
        .with_columns(bip_outcome = cl('events').replace_strict({'single':'single', 'double':'double',
                                                                 'triple':'triple', 'home_run':'home_run'},
                                                                default='out')))

outcome_map = {'single':1,'double':2,'triple':3,'home_run':4}
Xy = (df.filter(~cl('bip_outcome').eq('out'))
        .select('home_team', 'theta','launch_speed','launch_angle',cl('bip_outcome').replace(outcome_map))
        .collect()
        .drop_nulls().to_numpy())

X_tra,X_val,y_tra,y_val = train_test_split(Xy[:,:-1],Xy[:,-1],train_size=0.95)

base_model = CatBoostClassifier(loss_function='MultiClass',
                                langevin=True,iterations=4000,learning_rate=4e-2,
                                cat_features=[0])
outcome_given_hit = CalibratedClassifierCV(base_model, method='sigmoid')
outcome_given_hit.fit(X_tra, y_tra, eval_set=(X_val,y_val))

with open(model_dir / 'outcome-given-hit.pkl','wb') as f:
    pickle.dump(outcome_given_hit,f)

#outcome_given_hit = KNeighborsClassifier(n_neighbors=25,n_jobs=-1)
#outcome_given_hit.fit(Xy[:,:-1],Xy[:,-1].astype(str))

#with open('models/outcome-given-hit.pkl','wb') as f:
#    pickle.dump(outcome_given_hit,f)
#
