import numpy as np, polars as pl
cl = pl.col

''' Leverage index is the expected value of the absolute change in win probability
    conditional on the base out state (but then rescaled such that the mean = 1)
    Can be found empirically, but to avoid small sample issues I'll use a markov transition matrix
'''

# Load the win prob & transition matrices
wp = np.load('tables/p(win|inn,half,base,out,rdiff).npy')
T = np.load('tables/p(new_base,new_out,run|base,out).npy')

# each game state gets its own LI
li = np.zeros_like(wp)

# necessary for indexing shit
max_rdiff = (wp.shape[-1]-1)//2

# Build up the (raw) leverage index table, will be scaled later
for inn in range(10):
    for half in range(2):
        for bases in range(8):
            for outs in range(3):
                for rdiff in range(0,2*max_rdiff+1):
                    wp0 = wp[inn,half,bases,outs,rdiff]
                    next_states = np.stack(np.where(T[bases,outs])).T
                    p_next_states = T[bases,outs,*next_states.T]
                    next_states[:,-1] *= 1 if half else -1 # home team contribs positively to rdiff
                    next_states[:,-1]  = np.clip(next_states[:,-1]+rdiff,0,2*max_rdiff)
                    next_half          = np.where(next_states[:,1]==3,1-half,half)
                    next_inn           = np.where(next_half!=half,inn+1,inn).clip(0,8)
                    next_states[:, 1] %= 3
                    wp_next_states = wp[next_inn,next_half,*next_states.T]
                    li[inn,half,bases,outs,rdiff] = (abs(wp_next_states-wp0)*p_next_states).sum()

# Since the reported LI is scaled such that the average LI is 1,
# I'm opting to do this scaling empirically, by building up a 
# game state + li dataframe which I'll merge with retrosheet data and average
i, h, b, o, r = np.indices(li.shape, dtype=int)
i_flat = i.ravel()
h_flat = h.ravel()
b_flat = b.ravel()
o_flat = o.ravel()
r_flat = r.ravel()-max_rdiff
li_flat = li.ravel()
tmp_li_df = pl.LazyFrame({'inn_ind': i_flat,
                          'half_ind': h_flat,
                          'base_cd': b_flat,
                          'outs_when_up': o_flat,
                          'run_diff': r_flat,
                          'li': li_flat})

game_state = ['inn_ind','half_ind','base_cd','outs_when_up','run_diff']

df = pl.scan_parquet('data/2021-2024-full-retrosheet.parquet')
avg_li = (df.rename({i: i.lower() for i in df.collect_schema().names()})
            .with_columns(inn_ind = (cl('inn_ct')-1).clip(0,9),
                          half_ind = 'bat_home_id',
                          base_cd = 'start_bases_cd',
                          outs_when_up = 'outs_ct',
                          run_diff = cl('home_score_ct')-cl('away_score_ct'))
            .join(tmp_li_df,on=game_state)
            .select('li')
            .mean()
            .collect().item())

li_df = tmp_li_df.with_columns(li = cl('li')/avg_li).collect()
li_df.write_parquet('tables/leverage-index.parquet')

