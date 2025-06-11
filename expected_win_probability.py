import numpy as np, polars as pl, pickle as pkl
from catboost import CatBoostClassifier
from functools import lru_cache
from utils import transition_mapper
cl = pl.col

def xwp_given_outcome(inn,half,outs,base,rdiff,outcome,br_X,base_adv,wp_table):
    ''' Given the game state, a batted ball, & its outcome (out, single, double, etc.)
        Return the expected next game state win probability 
    '''
    max_rdiff = (wp_table.shape[-1]-1)//2
    codes = base_adv.classes_.astype(int)
    lead_base = base.bit_length()
    outcome_code = {'out': 0, 'single': 1, 'double': 2, 'triple': 3, 'home_run': 4}[outcome]
    match base.bit_count():
        case 0:
            lead_probs   = {None:1} # means they don't exist with a probabilty of 1.
            trail_probs  = {None:1}
            trail2_probs = {None:1}
        case 1:
            lead_probs   = dict(zip(codes,base_adv.predict_proba([outcome_code,'lead',lead_base,outs,*br_X])))
            trail_probs  = {None:1}
            trail2_probs = {None:1}
        case 2:
            trail_base   = (base & ~(1<<(lead_base-1))).bit_length()
            lead_probs   = dict(zip(codes,base_adv.predict_proba([outcome_code,'lead',lead_base,outs,*br_X])))
            trail_probs  = dict(zip(codes,base_adv.predict_proba([outcome_code,'trail',trail_base,outs,*br_X])))
            trail2_probs = {None:1}
        case 3:
            lead_probs   = dict(zip(codes,base_adv.predict_proba([outcome_code,'lead',3,outs,*br_X])))
            trail_probs  = dict(zip(codes,base_adv.predict_proba([outcome_code,'trail',2,outs,*br_X])))
            trail2_probs = dict(zip(codes,base_adv.predict_proba([outcome_code,'trail2',1,outs,*br_X])))
    xwp = 0 
    for aL,pL in lead_probs.items():
        for aT,pT in trail_probs.items():
            for aT2,pT2 in trail2_probs.items():
                prob = pL*pT*pT2
                if prob==0:
                    continue
                new_outs, new_base, runs = transition_mapper(outs, base, bat_event=outcome,
                                                             adv_codes=tuple(i for i in (aL, aT, aT2)
                                                                             if i is not None))
                new_rdiff = rdiff - runs if half==0 else rdiff + runs
                if (inn>=8 and half==1 and new_outs>2) and new_rdiff!=0:
                    new_wp = 1. if new_rdiff>0 else 0.
                    xwp += prob*new_wp
                    continue
                new_inn   = inn+1 if new_outs>2 and half>0 else inn
                new_inn   = min(new_inn,9)
                new_half  = half if new_outs<3 else 1-half
                new_base  = new_base if new_outs<3 else (2 if new_inn==9 else 0)
                new_outs  = new_outs if new_outs<3 else 0
                new_wp = wp_table[new_inn,new_half,new_base,new_outs,new_rdiff+max_rdiff]
                xwp += prob*new_wp
    return xwp

rwxwp_count = 0
def row_wise_xwp(row, base_adv, wp_table):
    global rwxwp_count
    rwxwp_count+=1
    inn   = row['inn_ind']
    half  = row['half_ind']
    outs  = row['outs_when_up']
    base  = row['base_cd']
    rdiff = row['run_diff']
    br_X  = np.array([row['theta'], row['launch_speed'], row['launch_angle']])
    p_out = row['out_prob']
    p_outcome_given_hit = np.array([row['p_1b'], row['p_2b'], row['p_3b'], row['p_hr']])
    p_1b, p_2b, p_3b, p_hr = (1-p_out)*p_outcome_given_hit
    xwp_of_outcome = lambda o: xwp_given_outcome(inn,half,outs,base,rdiff,o,br_X,base_adv,wp_table)
    xwp  = p_out*xwp_of_outcome('out')
    xwp += p_1b*xwp_of_outcome('single')
    xwp += p_2b*xwp_of_outcome('double')
    xwp += p_3b*xwp_of_outcome('triple')
    xwp += p_hr*xwp_of_outcome('home_run')
    print(rwxwp_count, inn, half, outs, base, rdiff, xwp)
    return xwp

