import numpy as np, polars as pl
from catboost import CatBoostClassifier
from utils import transition_mapper
cl = pl.col

''' Building up a transition probability matrix
    takes base,out states -> base,out,runs states
    I'm going to assume quite a lot of independence in outcomes, 
    much of which won't quite be true. This is an extensible framework at least
    factorized model: p(event); event ∈ {K,BB/HBP,in-play}
                      p(outcome | in-play)
                      p(runner adv | outcome,role,base,outs)
'''

def runner_adv_probs(base,outs,outcome_code,base_adv):
    codes = base_adv.classes_.astype(int)
    lead_base = base.bit_length()
    match base.bit_count():
        case 0:
            lead_probs   = {None:1} # means they don't exist with a probabilty of 1.
            trail_probs  = {None:1}
            trail2_probs = {None:1}
        case 1:
            lead_probs   = dict(zip(codes,base_adv.predict_proba([outcome_code,'lead',lead_base,outs])))
            trail_probs  = {None:1}
            trail2_probs = {None:1}
        case 2:
            trail_base   = (base & ~(1<<(lead_base-1))).bit_length()
            lead_probs   = dict(zip(codes,base_adv.predict_proba([outcome_code,'lead',lead_base,outs])))
            trail_probs  = dict(zip(codes,base_adv.predict_proba([outcome_code,'trail',trail_base,outs])))
            trail2_probs = {None:1}
        case 3:
            lead_probs   = dict(zip(codes,base_adv.predict_proba([outcome_code,'lead',3,outs])))
            trail_probs  = dict(zip(codes,base_adv.predict_proba([outcome_code,'trail',2,outs])))
            trail2_probs = dict(zip(codes,base_adv.predict_proba([outcome_code,'trail2',1,outs])))
    return lead_probs,trail_probs,trail2_probs

df = pl.read_parquet('data/2021-2024-full-retrosheet.parquet')
df = (df.rename({i: i.lower() for i in df.collect_schema().names()})
        .filter(cl('game_type').eq('R'),cl('bat_event_fl').eq('T'),cl('inn_ct')<9)
        .sort('game_id','inn_ct','bat_home_id','event_id')
        .with_columns(is_k = cl('event_cd').eq(3),
                      is_bb = cl('event_cd').is_in([14,15,16,17]),
                      is_out = cl('event_cd').is_in([2,3,19]),
                      is_1b = cl('event_cd').is_in([18,20]),
                      is_2b = cl('event_cd').eq(21),
                      is_3b = cl('event_cd').eq(22),
                      is_hr = cl('event_cd').eq(23))
        .with_columns(is_inplay = ~(cl('is_k') | cl('is_bb'))))

p_events = df.select('is_k','is_bb','is_inplay').mean().to_numpy().squeeze()
p_outcomes = df.filter('is_inplay').select('is_out','is_1b','is_2b','is_3b','is_hr').mean().to_numpy().squeeze()

# Transition probability matrix
# 8 start base states, 3 start out states, 
# 8 end base states, 4 end out states, 5 run scoring states
T = np.zeros((8,3,8,4,5))

# Baserunner advancement model
base_adv = CatBoostClassifier().load_model('models/baserunner-advancement-no-sc.cbm')

for bases in range(8):
    for outs in range(3):
        for event,p_event in enumerate(p_events):
            match event:
                case 0: # strikeout
                    new_outs = outs+1
                    new_bases = bases if new_outs<3 else 0
                    runs = 0
                    T[bases,outs,new_bases,new_outs,runs] += p_event
                case 1: # walk
                    new_outs,new_bases,runs = transition_mapper(outs,bases,'walk',[])
                    T[bases,outs,new_bases,new_outs,runs] += p_event
                case 2: # BIP
                    for outcome,p_outcome in enumerate(p_outcomes):
                        lead_probs,trail_probs,trail2_probs = runner_adv_probs(bases,outs,outcome,base_adv)
                        for aL,pL in lead_probs.items():
                            for aT,pT in trail_probs.items():
                                for aT2,pT2 in trail2_probs.items():
                                    p_adv = pL*pT*pT2
                                    if p_adv==0:
                                        continue
                                    adv_codes = tuple(i for i in (aL, aT, aT2) if i is not None)
                                    outcomes = ['out','single','double','triple','home_run']
                                    new_outs, new_bases, runs = transition_mapper(outs,bases,outcomes[outcome],adv_codes)
                                    new_bases = new_bases if new_outs<3 else 0
                                    new_outs = min(3,new_outs)
                                    T[bases,outs,new_bases,new_outs,runs] += p_adv*p_outcome*p_event

# save it
np.save('tables/p(new_base,new_out,run|base,out).npy',T)

