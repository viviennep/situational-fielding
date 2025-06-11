import numpy as np, polars as pl
cl = pl.col

''' Game state is inn, half, outs, base, run diff
    I'll start with 9 innings and a fixed extra inning win prob
    I'm only considering run diffs of -14 to 14
    Everything is in the home team's perspective
'''

# Load in run distribution tables
pr  = np.load('tables/p(runs|base,out).npy')
pr9 = np.load('tables/p(runs|base,out,need1).npy')
pr9 = (pr+pr9)/2

max_rdiff = 30
max_rinn  = 14

# Find home team extra innings win% (turned out to be slightly under .500, huh)
p_xiw = (pl.scan_parquet('data/2021-2024-full-retrosheet.parquet')
           .filter(cl('INN_CT')>9,cl('GAME_END_FL').eq('T'))
           .with_columns(pre_rdiff =cl('HOME_SCORE_CT')-cl('AWAY_SCORE_CT'),
                         play_rdiff=pl.when(cl('BAT_HOME_ID').eq(1))
                                      .then(cl('EVENT_RUNS_CT'))
                                      .otherwise(-cl('EVENT_RUNS_CT')))
           .select(((cl('pre_rdiff')+cl('play_rdiff'))>0).mean())
           .collect()
           .item())

# Build table, set home team leading in bot 9 situations to 1.
wp = np.zeros((9,2,8,3,max_rdiff*2+1),dtype='f')
wp[-1,1,...,max_rdiff+1:] = 1.

# fill in rest of bot 9, which is special
# p(win) = p(walkoff) + p(go to extras)*p(win in extras)
for o in range(3):
    for b in range(8):
        for rdiff in range(-max_rdiff,1):
            #cur_pr = pr
            cur_pr = pr9 if rdiff in [-1,0] else pr
            p_walkoff = cur_pr[b,o,-rdiff+1:].sum()
            p_extras  = cur_pr[b,o,-rdiff:-rdiff+1].sum() # syntax manip
            p_win     = p_walkoff + p_extras*p_xiw
            wp[-1,1,b,o,max_rdiff+rdiff] = p_win

# in general, for top of inning i, clipping run diff within range:
# wp = sum( p(r|b,o)*wp(i,1,b,o,clip(rdiff-r,-max_rdiff,max_rdiff)) )
# could build matrix wp(i,1,b,o,clip(rdiff-r,-max_rdiff,max_rdiff)) for each rdiff & r
# size (2*max_rdiff+1, max_rinn), result of product with pr[b,o] (dim r) would be wp for each rdiff 
rdiffs   = np.arange(-max_rdiff,max_rdiff+1)
rs       = np.arange(max_rinn+1)
rdr      = np.subtract.outer(rdiffs,rs).clip(-max_rdiff,max_rdiff)
rdr_ind  = rdr+max_rdiff
rdr_inds = np.stack((rdr_ind,max_rdiff*2-rdr_ind[::-1]))

for i in range(8,-1,-1):
    for h in [1,0]:
        if h==1 and i==8:
            continue
        for o in range(3):
            for b in range(8):
                wp[i,h,b,o] = wp[i+h,1-h,0,0,rdr_inds[h]]@pr[b,o]
                if i==8: # change to the "need a run" run dist
                    wp9 = wp[i+h,1-h,0,0,rdr_inds[h]]@pr9[b,o]
                    wp[i,h,b,o,max_rdiff-1:max_rdiff+1] = wp9[max_rdiff-1:max_rdiff+1]

# the bottom of extras is the same as the bottom of the 9th,
# the top is different because the next inning start state 
# begins with a runner on second
# so the wp matrix is wp[8,1,2,0,rdr_inds[0]]
extras_wp = wp[-1].copy()
for o in range(3):
    for b in range(8):
        extras_wp[0,b,o] = wp[8,1,2,0,rdr_inds[0]]@pr[b,o]
        # also account for the "need 1 run" case
        wp9 = wp[8,1,2,0,rdr_inds[0]]@pr9[b,o]
        extras_wp[0,b,o,max_rdiff-1:max_rdiff+1] = wp9[max_rdiff-1:max_rdiff+1]

wp = np.concatenate((wp,extras_wp[None,...]),axis=0)

np.save('tables/p(win|inn,half,base,out,rdiff).npy',wp)

i, h, b, o, r = np.indices(wp.shape, dtype=int)
i_flat = i.ravel()
h_flat = h.ravel()
b_flat = b.ravel()
o_flat = o.ravel()
r_flat = r.ravel()-max_rdiff
wp_flat = wp.ravel()

wp_df = pl.DataFrame({'inn_ind': i_flat,
                      'half_ind': h_flat,
                      'base_cd': b_flat,
                      'outs_when_up': o_flat,
                      'run_diff': r_flat,
                      'wp': wp_flat})
wp_df.write_parquet('tables/p(win|inn,half,base,out,rdiff).parquet')

