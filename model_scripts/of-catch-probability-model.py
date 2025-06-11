import numpy as np, polars as pl, matplotlib.pyplot as plt, pickle as pkl
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
cl = pl.col

of_plays = pl.read_parquet('data/2021-2024-of-plays-with-wall.parquet')

features = ['dist','angle','hang_time','wall_dist_start',
            'wall_dist_land','wall_dist_ball_dir','wall_min_dist','wall_height']

X    = of_plays.select(features).to_numpy()
wall = of_plays.select('wall').to_numpy().squeeze()
y    = of_plays['out'].to_numpy()

X_tra,X_val,wall_tra,wall_val,y_tra,y_val = train_test_split(X,wall,y,train_size=0.80)

base_model = CatBoostClassifier(loss_function='Focal:focal_alpha=0.7;focal_gamma=5.0',
                                langevin=True,iterations=5000,learning_rate=1e-2)
catch_prob = CalibratedClassifierCV(base_model, method='sigmoid')
catch_prob.fit(X_tra, y_tra, eval_set=(X_val,y_val))

with open('models/catch-prob.pkl','wb') as f:
    pkl.dump(catch_prob,f)

''' For quick validation
'''
#pred = catch_prob.predict_proba(X)[:,-1]
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
