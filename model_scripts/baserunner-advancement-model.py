import numpy as np, polars as pl
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier 
cl = pl.col

df = pl.read_parquet('data/2021-2024-retro-and-statcast.parquet')

# Rotate & scale stringer hit location data then calc spray angle
scale = 2.495671
df = (df.with_columns(hc_x_ft=scale*( cl('hc_x')-125.42),
                      hc_y_ft=scale*(-cl('hc_y')+198.27))
        .with_columns(theta=pl.arctan2('hc_x_ft','hc_y_ft')))

# Identify whether each runner is a lead or trail runner (or with bases loaded, double trailer)
N = None
lead_runner_base   = { 0b000: N, 0b001: 1, 0b010: 2, 0b011: 2, 0b100: 3, 0b101: 3, 0b110: 3, 0b111: 3}
trail_runner_base  = { 0b000: N, 0b001: N, 0b010: N, 0b011: 1, 0b100: N, 0b101: 1, 0b110: 2, 0b111: 2}
trail2_runner_base = { 0b000: N, 0b001: N, 0b010: N, 0b011: N, 0b100: N, 0b101: N, 0b110: N, 0b111: 2}

# First map all scoring retrosheet runner destination codes to 3, (which for me represents all scoring)
# Then identify lead & trail runners and calculate their advancements
df = (df.with_columns(run1_dest_id = pl.when(cl('run1_dest_id').is_in([4,5,6,7])).then(4).otherwise('run1_dest_id'),
                      run2_dest_id = pl.when(cl('run2_dest_id').is_in([4,5,6,7])).then(4).otherwise('run2_dest_id'),
                      run3_dest_id = pl.when(cl('run3_dest_id').is_in([4,5,6,7])).then(4).otherwise('run3_dest_id'))
        .with_columns(lead_runner_base   = cl('start_bases_cd').replace(lead_runner_base),
                      trail_runner_base  = cl('start_bases_cd').replace(trail_runner_base),
                      trail2_runner_base = cl('start_bases_cd').replace(trail2_runner_base))
        .with_columns(lead_runner_dest   = pl.when(cl('lead_runner_base').eq(1))
                                             .then('run1_dest_id')
                                             .when(cl('lead_runner_base').eq(2))
                                             .then('run2_dest_id')
                                             .when(cl('lead_runner_base').eq(3))
                                             .then('run3_dest_id')
                                             .otherwise(None),
                      trail_runner_dest  = pl.when(cl('trail_runner_base').eq(1))
                                             .then('run1_dest_id')
                                             .when(cl('trail_runner_base').eq(2))
                                             .then('run2_dest_id')
                                             .otherwise(None),
                      trail2_runner_dest = pl.when(cl('trail2_runner_base').eq(1))
                                             .then('run1_dest_id')
                                             .otherwise(None))
        .with_columns(lead_runner_adv    = pl.when(cl('lead_runner_dest').eq(4))
                                             .then(3)
                                             .otherwise(cl('lead_runner_dest')-cl('lead_runner_base'))
                                             .clip(-1),
                      trail_runner_adv   = pl.when(cl('trail_runner_dest').eq(4))
                                             .then(3)
                                             .otherwise(cl('trail_runner_dest')-cl('trail_runner_base'))
                                             .clip(-1),
                      trail2_runner_adv  = pl.when(cl('trail2_runner_dest').eq(4))
                                             .then(3)
                                             .otherwise(cl('trail2_runner_dest')-cl('trail2_runner_base'))
                                             .clip(-1)))

# Building up structs to later explode such that each row corresponds to a runner (rather than a pitch)
lead_struct   = pl.struct([cl('lead_runner_adv').alias('adv'),
                           pl.lit('lead').alias('runner_role'),
                           cl('lead_runner_base').alias('base')])
trail_struct  = pl.struct([cl('trail_runner_adv').alias('adv'),
                           pl.lit('trail').alias('runner_role'),
                           cl('trail_runner_base').alias('base')])
trail2_struct = pl.struct([cl('trail2_runner_adv').alias('adv'),
                           pl.lit('trail2').alias('runner_role'),
                           cl('trail2_runner_base').alias('base')])

# Turn event types into bbe types
bbe_type_codes = {'double_play': 0,
                  'grounded_into_double_play': 0,
                  'field_out': 0,
                  'fielders_choice_out': 0,
                  'single': 1,
                  'sac_bunt': 0,
                  'fielders_choice': 0,
                  'triple_play': 0,
                  'double': 2,
                  'sac_bunt_double_play': 0,
                  'field_error': 0,
                  'catcher_interf': 1,
                  'sac_fly': 0,
                  'sac_fly_double_play': 0,
                  'home_run': 4,
                  'force_out': 0,
                  'triple': 3}

# Explode the df into 1-row-per-runner 
df = (df.filter(~cl('start_bases_cd').eq(0),cl('launch_speed').is_not_null())
        .with_columns(runner=pl.concat_list([lead_struct,trail_struct,trail2_struct]),
                      run_diff=cl('bat_score')-cl('fld_score'),
                      bbe_type=cl('events').replace_strict(bbe_type_codes,return_dtype=pl.Int8))
        .explode('runner')
        .unnest('runner')
        .filter(cl('adv').is_not_null()))

# Extract features for model & train
features = ['bbe_type','runner_role','base','outs_when_up','theta','launch_speed','launch_angle']
Xy = df.select(features+['adv']).to_numpy()
X = Xy[:,:-1]
y = Xy[:,-1].astype(str) # I don't want it to treat the results numerically
Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.2)
br_adv = CatBoostClassifier(cat_features=[0,1],
                            learning_rate=4e-2,
                            iterations=2500)
br_adv.fit(Xtr,ytr,eval_set=(Xva,yva))
br_adv.save_model('models/baserunner-advancement.cbm')

# Train reduced model for markov transition & leverage index calculation
#features = ['bbe_type','runner_role','base','outs_when_up']
#Xy = df.select(features+['adv']).to_numpy()
#X = Xy[:,:-1]
#y = Xy[:,-1].astype(str) # I don't want it to treat the results numerically
#Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.2)
#br_adv = CatBoostClassifier(cat_features=[0,1],
#                            learning_rate=1e-2,
#                            iterations=2500)
#br_adv.fit(Xtr,ytr,eval_set=(Xva,yva))
#br_adv.save_model('models/baserunner-advancement-no-sc.cbm')

