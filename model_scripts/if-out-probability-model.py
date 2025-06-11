import numpy as np, polars as pl, matplotlib.pyplot as plt, pickle as pkl
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
cl = pl.col

''' WICKED BAD !! NOT INTENDED TO BE TAKEN SERIOUSLY !!
    No sprint speed data, just predicting out from BBE characteristics
    In practice this performs so terribly that it's unusable 
    just scaffolding for future improvement
'''

of_plays = pl.scan_parquet('data/2021-2024-of-plays-with-wall.parquet')
if_plays = (pl.scan_parquet('data/2021-2024-sc-with-playid.parquet')
              .filter(cl('launch_speed').is_not_null(),cl('hc_x').is_not_null())
              .join(of_plays,on='play_id',how='anti')
              .with_columns(hc_x_ft = 2.495671*( cl('hc_x')-125.42), 
                            hc_y_ft = 2.495671*(-cl('hc_y')+198.27))
              .with_columns(theta = pl.arctan2('hc_x_ft','hc_y_ft'),
                            bip_outcome = cl('events').replace_strict({'single':'single', 'double':'double',
                                                                       'triple':'triple', 'home_run':'home_run'},
                                                                      default='out'),
                            if_fielding_alignment=pl.when(cl('if_fielding_alignment').is_not_null())
                                                    .then('if_fielding_alignment')
                                                    .otherwise(pl.lit('Unknown')))
              .collect())

features = ['if_fielding_alignment','stand','theta','launch_speed','launch_angle']

X = if_plays.select(features).to_numpy()
y = if_plays.select(cl('bip_outcome').eq('out')).to_numpy().squeeze()

X_tra,X_val,y_tra,y_val = train_test_split(X,y,train_size=0.80)

base_model = CatBoostClassifier(loss_function='Focal:focal_alpha=0.7;focal_gamma=5.0',
                                langevin=True,iterations=4000,learning_rate=4e-2,
                                cat_features=[0,1])
out_prob = CalibratedClassifierCV(base_model, method='sigmoid')
out_prob.fit(X_tra, y_tra, eval_set=(X_val,y_val))

with open('models/out-prob.pkl','wb') as f: pkl.dump(out_prob,f)

''' For quick validation
'''
#pred = out_prob.predict_proba(X)[:,-1]
#
#def calc_calib_curve(p,y,nbins=25):
#    bins = np.linspace(0,1,nbins)[1:]
#    binned_p = np.digitize(p,bins)
#    real = np.bincount(binned_p,weights=y)/np.bincount(binned_p)
#    pred = np.bincount(binned_p,weights=p)/np.bincount(binned_p)
#    return pred,real
#
#f,ax = plt.subplots()
#ax.plot(*calc_calib_curve(pred,y))
#ax.plot([0,1],[0,1],c='crimson',ls='-',lw=1,zorder=-1)
#ax.set_aspect('equal')
#plt.show()

