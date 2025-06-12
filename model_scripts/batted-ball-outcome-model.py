import numpy as np, polars as pl, pickle, pathlib, optuna, lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
cl = pl.col
data_dir = pathlib.Path('../data')
model_dir = pathlib.Path('../models')

#df = (pl.scan_parquet(data_dir / '2021-2024-sc-with-playid.parquet')
df = (pl.scan_parquet(data_dir / 'daily_data', hive_partitioning=True)
        .with_columns(bip_outcome = cl('events').replace_strict({'single':'single', 
                                                                 'double':'double',
                                                                 'triple':'triple', 
                                                                 'home_run':'home_run'},
                                                                default='out')))

outcome_map = {'single':1,'double':2,'triple':3,'home_run':4}
Xy = (df.filter(~cl('bip_outcome').eq('out'),)
        .select('home_team','hc_dist','theta','launch_speed','launch_angle',cl('bip_outcome').replace(outcome_map))
        .collect()
        .drop_nulls().to_numpy())

X_tra,X_val,y_tra,y_val = train_test_split(Xy[:,:-1],Xy[:,-1],train_size=0.95)

def objective(trial):
    params = {'loss_function': 'MultiClass',
              'iterations': 3000,
              'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
              'depth': trial.suggest_int('depth', 4, 10),
              'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-2, 100.0),
              'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
              'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
              'border_count': trial.suggest_int('border_count', 32, 255),
              'langevin': True,
              'verbose': True}
    pruning_callback = optuna.integration.CatBoostPruningCallback(trial, 'MultiClass')
    model = CatBoostClassifier(**params)
    model.fit(X_tra, y_tra,
              cat_features=[0],
              eval_set=(X_val,y_val),
              early_stopping_rounds=50,
              callbacks=[pruning_callback])
    preds = model.predict_proba(X_val)
    loss = log_loss(y_val, preds)
    return loss

study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
)
study.optimize(objective, n_trials=20, n_jobs=1)

best_params = {'loss_function': 'MultiClass',
               'learning_rate': 0.017243965252688496, 
               'depth': 9, 
               'l2_leaf_reg': 0.0117142176647651, 
               'random_strength': 1.5777029282158956, 
               'bagging_temperature': 0.021618783519340084, 
               'border_count': 197,
               'langevin': True}


outcome_given_hit = CatBoostClassifier(**best_params,
                                       task_type='GPU',
                                       iterations=4000,
                                       cat_features=[0])
outcome_given_hit.fit(X_tra, y_tra, eval_set=(X_val,y_val))

with open(model_dir / 'outcome-given-hit.pkl','wb') as f:
    pickle.dump(outcome_given_hit,f)


''' For quick validation
'''
#X = Xy[:,:-1]
#y = Xy[:,-1]
#pred = outcome_given_hit.predict_proba(X)
#
#def calc_calib_curve(p,y,nbins=25):
#    bins = np.linspace(0,1,nbins)[1:]
#    binned_p = np.digitize(p,bins)
#    real = np.bincount(binned_p,weights=y)/np.bincount(binned_p)
#    pred = np.bincount(binned_p,weights=p)/np.bincount(binned_p)
#    return pred,real
#
#f,ax = plt.subplots(2,2)
#ax[0,0].plot(*calc_calib_curve(pred[:,0],y=='1'))
#ax[0,0].plot([0,1],[0,1],c='crimson',ls='-',lw=1,zorder=-1)
#ax[1,0].plot(*calc_calib_curve(pred[:,1],y=='2'))
#ax[1,0].plot([0,1],[0,1],c='crimson',ls='-',lw=1,zorder=-1)
#ax[0,1].plot(*calc_calib_curve(pred[:,2],y=='3'))
#ax[0,1].plot([0,1],[0,1],c='crimson',ls='-',lw=1,zorder=-1)
#ax[1,1].plot(*calc_calib_curve(pred[:,3],y=='4'))
#ax[1,1].plot([0,1],[0,1],c='crimson',ls='-',lw=1,zorder=-1)
#plt.setp(ax,aspect='equal')
#plt.show()
#
#
