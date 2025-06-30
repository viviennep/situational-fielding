import numpy as np, polars as pl, matplotlib.pyplot as plt, pickle as pkl
import xgboost as xgb, optuna
from xgboost import XGBClassifier
from optuna.integration import XGBoostPruningCallback 
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
ipw = (ipw+3)/4

def focal_loss(alpha: float = 0.70, gamma: float = 5.0):
    def _objective(pred, dmat):
        y   = dmat.get_label()
        p   = 1/(1+np.exp(-pred))              # sigmoid
        pt  = np.where(y==1, p, 1-p)           # p_t in the paper
        at  = np.where(y==1, alpha, 1-alpha)   # α_t
        g   = at*(1-pt)**gamma                 # common factor
        grad = g*(gamma*pt*np.log(pt + 1e-12) + 1)*(p - y)
        hess = g*((gamma*pt*(1-pt)*(gamma*np.log(pt + 1e-12) + 1))
                  +(p - y)*(gamma*(1-2*p)))    # exact second derivative
        return grad, hess

    def _metric(pred, dmat):
        y = dmat.get_label()
        p = 1/(1 + np.exp(-pred))
        pt = np.where(y==1, p, 1-p)
        fl = -(alpha*y + (1-alpha)*(1-y))*(1-pt)**gamma*np.log(pt + 1e-12)
        return "focal_loss", float(np.mean(fl))

    return _objective, _metric

obj,met = focal_loss(0.7,2.0)

def objective(trial):
    params = {
        'device'           : 'cuda',
        'tree_method'      : 'hist',
        'predictor'        : 'gpu_predictor',
        'monotone_constraints': '(-1,1,0,0,0,0)',
        'learning_rate'    : trial.suggest_float('learning_rate', 1e-5, 0.15, log=True),
        'max_depth'        : trial.suggest_int('max_depth', 3, 10),
        'min_child_weight' : trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_bin'          : trial.suggest_int('max_bin', 256, 1024, step=128)
    }
    dtrain = xgb.DMatrix(X, label=y, weight=ipw)
    cv_res = xgb.cv(
        params,
        dtrain,
        obj                    = obj,
        custom_metric          = met,
        nfold                  = 5,
        num_boost_round        = 8000,
        early_stopping_rounds  = 50,
        verbose_eval           = True,
        stratified             = True,
        callbacks=[XGBoostPruningCallback(trial, 'test-focal_loss')],
    )
    trial.set_user_attr('n_boosted_rounds', len(cv_res))
    return cv_res['test-focal_loss-mean'].values[-1]    # minimise log-loss

# run the optimisation
study = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(),
                            pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=200)

best_params = study.best_trial.params
best_n_boosted_rounds = study.best_trial.user_attrs['n_boosted_rounds']

#best_params = {
#    'learning_rate'       : 0.0005499746110327788,
#    'max_depth'           : 5,
#    'min_child_weight'    : 0.19556308660449387,
#    'subsample'           : 0.809814775821074,
#    'colsample_bytree'    : 0.7327700975317445,
#    'reg_alpha'           : 0.1092249847144697,
#    'reg_lambda'          : 8.481222978866436,
#    'max_bin'             : 896,
#}
#
#best_n_boosted_rounds = 1130

X_tra,X_val,y_tra,y_val,w_tra,w_val = train_test_split(X,y,ipw,train_size=0.80)

base_model = XGBClassifier(
    **best_params,
    monotone_constraints = '(-1,1,0,0,0,0)',
    n_boosted_rounds     = best_n_boosted_rounds,
    objective            = obj,
    custom_metric        = met
)

catch_prob = CalibratedClassifierCV(base_model, method='sigmoid')
catch_prob.fit(
    X_tra, 
    y_tra, 
    sample_weight=w_tra,
    eval_set=[(X_val,y_val)], 
)

#'''
#
#base_model = CatBoostClassifier(loss_function='Focal:focal_alpha=0.7;focal_gamma=5.0',
#                                langevin=True,iterations=5000,learning_rate=1e-2)
#catch_prob.fit(X_tra, y_tra, eval_set=(X_val,y_val))
#
#with open('models/catch-prob.pkl','wb') as f:
#    pkl.dump(catch_prob,f)
#'''

''' For quick validation
'''

pred = catch_prob.predict_proba(X)[:,-1]

def calc_calib_curve(p,y,nbins=10):
    bins = np.linspace(0,1,nbins)[1:]
    binned_p = np.digitize(p,bins)
    real = np.bincount(binned_p,weights=y)/np.bincount(binned_p)
    pred = np.bincount(binned_p,weights=p)/np.bincount(binned_p)
    return pred,real

f,ax = plt.subplots()
ax.plot(*calc_calib_curve(pred,y))
ax.plot(*calc_calib_curve(normal_plays.select('catch_rate').to_numpy().squeeze(),y))
ax.plot([0,1],[0,1],c='crimson',ls='-',lw=1,zorder=-1)
ax.set_aspect('equal')
plt.show()


