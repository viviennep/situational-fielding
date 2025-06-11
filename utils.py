import numpy as np, polars as pl, requests

headers = {'User-Agent': 'Mozilla/5.0'}

def transition_mapper(outs, base, bat_event, adv_codes):
    ''' Given runner advancement codes & the current base-out state, 
        return the next base-out state (& any runs accumulated in the transition)
    '''
    outs, base, runs = outs, base, 0
    bases_present = [b for b in (3, 2, 1) if base & (1 << (b - 1))]
    for cur, code in zip(bases_present, adv_codes):
        match code:
            case -1:
                dest, add_runs, add_outs = None, 0, 1
            case 3:
                dest, add_runs, add_outs = None, 1, 0
            case adv if adv in (0, 1, 2):
                dest = cur + adv
                if dest >= 4:
                    dest, add_runs, add_outs = None, 1, 0
                else:
                    dest, add_runs, add_outs = dest, 0, 0
            case _:
                dest, add_runs, add_outs = cur, 0, 0
        runs += add_runs
        outs += add_outs
        base &= ~(1 << (cur - 1))
        if dest:
            base |= (1 << (dest - 1))
    match bat_event:
        case 'out':
            outs += 1
        case 'single':
            base |= 0b001
        case 'double':
            base |= 0b010
        case 'triple':
            base |= 0b100
        case 'home_run':
            runs += base.bit_count() + 1
            base = 0
        case 'walk':
            ori_base = base
            for b in (3,2,1):
                if ori_base & (1<<(b-1)): # is base occupied
                    if b==1 or all(ori_base & (1<<(i-1)) for i in range(1,b)): # is base forced
                        base &= ~(1<<(b-1)) # remove from cur base
                        if b==3:
                            runs += 1
                        else:
                            base |= 1<<b 
            base |= 0b001
    if outs >= 3:
        outs = 3
        base = 0
    return outs, base, runs

