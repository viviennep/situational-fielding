import numpy as np, polars as pl, matplotlib.pyplot as plt, pickle as pkl
import catboost, xgboost as xgb, optuna
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from optuna.integration import CatBoostPruningCallback 
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from scipy.stats import ecdf
from scipy.interpolate import make_smoothing_spline
cl = pl.col

of_plays = pl.read_parquet('../data/2021-2024-of-plays-with-wall.parquet')

# Including angle encoded as sin & cos to enforce periodicity
# Add a flag for whether play is wall ball or not
of_plays = of_plays.with_columns(sin_theta=cl('angle').radians().sin(),
                                 cos_theta=cl('angle').radians().cos(),
                                 wall_ball=(cl('wall_dist_hit')-cl('hit_dist')<0))

wall_balls = of_plays.filter('wall_ball')
normal_plays = of_plays.filter(~cl('wall_ball'))

# Features for normal plays
features = ['dist','hang_time',
            'sin_theta','cos_theta',
            'wall_dist_land','wall_min_dist']
X    = normal_plays.select(features).to_numpy()
y    = normal_plays['out'].to_numpy()

# Create simple model to rewiegh data via inverse-probs
simple_model = XGBClassifier(
    objective = 'binary:logistic',
    eval_metric='logloss',
)
simple_model.fit(X,y)
preds = simple_model.predict_proba(X)[:,-1]
res = ecdf(preds)
cdf = make_smoothing_spline(res.cdf.quantiles,res.cdf.probabilities,lam=1e-10)
pdf = cdf.derivative()
qs  = (preds.argsort().argsort()+1)/(preds.size+1)
ip  = 1/pdf(qs)
ipw = ip/ip.mean()
ipw = (ipw+2)/3

cv = StratifiedKFold(n_splits=3, shuffle=True)

loss_str = 'Focal:focal_alpha=0.5;focal_gamma=2.0'

def create_objective(X,y,ipw,iterations=7000):
    def objective(trial):
        params = {
            'loss_function'        : loss_str,
            'eval_metric'          : loss_str,
            'iterations'           : iterations,
            'learning_rate'        : trial.suggest_float('learning_rate', 1e-4, 0.15, log=True),
            'depth'                : trial.suggest_int('depth', 3, 12),
            'l2_leaf_reg'          : trial.suggest_float('l2_leaf_reg', 1e-6, 10.0, log=True),
            'subsample'            : trial.suggest_float('subsample', 0.6, 1.0),
            'monotone_constraints' : [-1, 1, 0, 0, 0, 0],
            'langevin'             : True,
            'diffusion_temperature': trial.suggest_float('diffusion_temperature',500,20000),
            'bagging_temperature'  : trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'border_count'         : trial.suggest_int('border_count', 32, 255),
        }
        fold_losses = [] 
        for step, (train_idx, valid_idx) in enumerate(cv.split(X, y), 1):
            train_pool = Pool(X[train_idx], y[train_idx], weight=ipw[train_idx])
            valid_pool = Pool(X[valid_idx], y[valid_idx], weight=ipw[valid_idx])
            model = CatBoostClassifier(**params)
            cb_prune = CatBoostPruningCallback(trial, loss_str)
            model.fit(
                train_pool,
                eval_set               = valid_pool,
                early_stopping_rounds  = 50,
                callbacks              = [cb_prune],
            )
            fold_losses.append(model.best_score_['validation'][loss_str])
            trial.report(np.mean(fold_losses), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        mean_loss = float(np.mean(fold_losses))
        trial.set_user_attr('n_estimators', 
                            int(np.mean([model.get_best_iteration() for _ in fold_losses])))
        return mean_loss
    return objective

# run the optimisation
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner())
#study.optimize(create_objective(X,y,ipw), n_trials=80)

#best_params = study.best_trial.params
#best_n_boosted_rounds = study.best_trial.user_attrs['n_estimators']

#best_params = {
#   'learning_rate'      : 0.01416207245436363,
#   'depth'              : 9,
#   'l2_leaf_reg'        : 4.266944680379848,
#   'subsample'          : 0.7728146471556169,
#   'diffusion_temperature': 8811.362660334366,
#   'bagging_temperature': 0.8738025313208493,
#   'border_count'       : 221,
#}
best_params = {
    'learning_rate': 0.01397614097512464, 
    'depth': 9, 
    'l2_leaf_reg': 0.08429451114080028, 
    'subsample': 0.755132389766684, 
    'diffusion_temperature': 3998.451504742423, 
    'bagging_temperature': 0.694641367398454, 
    'border_count': 231
}

X_tra,X_val,y_tra,y_val,w_tra,w_val = train_test_split(X,y,ipw,train_size=0.80)

normal_base_model = CatBoostClassifier(
    **best_params,
    loss_function=loss_str,
    eval_metric=loss_str,
    langevin=True,
    iterations=1100,
    monotone_constraints=[-1, 1, 0, 0, 0, 0],
)

normal_catch_prob = CalibratedClassifierCV(normal_base_model, method='isotonic')
normal_catch_prob.fit(
    X_tra, 
    y_tra, 
    sample_weight=w_tra,
    eval_set=[(X_val,y_val)], 
)

with open('../models/normal-catch-prob.pkl','wb') as f:
    pkl.dump(normal_catch_prob,f)

# Features for normal plays
features = ['dist','hang_time',
            'sin_theta','cos_theta',
            'wall_dist_land','wall_min_dist',
            'wall_height','wall_height_hit',
            'hit_dist']
X    = wall_balls.select(features).to_numpy()
y    = wall_balls['out'].to_numpy()

study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner())
obj = create_objective(X, y, np.ones_like(y))

#study.optimize(obj, n_trials=80)

#best_params = study.best_trial.params
best_params = {
    'learning_rate': 0.012295594745733555, 
    'depth': 6, 
    'l2_leaf_reg': 0.00011579923820249005, 
    'subsample': 0.9542878491471074, 
    'diffusion_temperature': 15797.635720873857, 
    'bagging_temperature': 0.8401956829960678, 
    'border_count': 230
}

X_tra,X_val,y_tra,y_val = train_test_split(X,y,train_size=0.80)

wall_base_model = CatBoostClassifier(
    **best_params,
    loss_function=loss_str,
    eval_metric=loss_str,
    langevin=True,
    iterations=2100,
    monotone_constraints=[-1, 1, 0, 0, 0, 0, -1, -1, 0],
)

wall_catch_prob = CalibratedClassifierCV(wall_base_model, method='isotonic')
wall_catch_prob.fit(
    X_tra, 
    y_tra, 
    eval_set=[(X_val,y_val)], 
)

with open('../models/wall-catch-prob.pkl','wb') as f:
    pkl.dump(wall_catch_prob,f)

''' For quick validation
'''

pred = wall_catch_prob.predict_proba(X)[:,-1]
pred = normal_catch_prob.predict_proba(X)[:,-1]

def calc_calib_curve(p,y,nbins=10):
    bins = np.linspace(0,1,nbins)[1:]
    binned_p = np.digitize(p,bins)
    real = np.bincount(binned_p,weights=y)/np.bincount(binned_p)
    pred = np.bincount(binned_p,weights=p)/np.bincount(binned_p)
    return pred,real

f,ax = plt.subplots()
ax.plot(*calc_calib_curve(pred,y))
#ax.plot(*calc_calib_curve(wall_balls.select('catch_rate').to_numpy().squeeze(),y))
ax.plot(*calc_calib_curve(normal_plays.select('catch_rate').to_numpy().squeeze(),y))
ax.plot([0,1],[0,1],c='crimson',ls='-',lw=1,zorder=-1)
ax.set_aspect('equal')
plt.show()

