import numpy as np, polars as pl, matplotlib.pyplot as plt
cl = pl.col; rng = np.random.default_rng()

# Read in the data, 
# get regular seasons batting events before the 9th inning
# calculate the rest of the inning runs
# find the distribution of runs scored given base out state
df = pl.scan_parquet('data/2021-2024-full-retrosheet.parquet')
df = (df.rename({i: i.lower() for i in df.collect_schema().names()})
        .filter(cl('game_type').eq('R'),cl('bat_event_fl').eq('T'),cl('inn_ct')<9)
        .sort('game_id','inn_ct','bat_home_id','event_id')
        .with_columns(rest_inn_runs=cl('fate_runs_ct')+cl('event_runs_ct'))
        .group_by('start_bases_cd','outs_ct','rest_inn_runs')
        .agg(pl.len().alias('n'))
        .sort('outs_ct','start_bases_cd','rest_inn_runs')
        .with_columns(p=cl('n')/(cl('n').sum().over('start_bases_cd','outs_ct')))
        .collect())

# Build up the numpy matrix from the dataframe
m = np.zeros((8,3,15))
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        ind,p = df.filter(cl('start_bases_cd').eq(i),cl('outs_ct').eq(j)).select('rest_inn_runs','p').to_numpy().T
        m[i,j,ind.astype(int)] = p

np.save('tables/p(runs|base,out).npy',m)

# Do the same thing but for 9th inning or greater, 
# when the batting team needs only 1 run to tie or win the game
# (I'm assuming that teams behave differently in these scenarios)
df = pl.scan_parquet('data/2021-2024-full-retrosheet.parquet')
df = (df.rename({i: i.lower() for i in df.collect_schema().names()})
        .filter(cl('game_type').eq('R'),cl('bat_event_fl').eq('T'),cl('inn_ct')>=9)
        .sort('game_id','inn_ct','bat_home_id','event_id')
        .with_columns(bat_deficit=cl('start_bat_score_ct')-cl('start_fld_score_ct'))
        .filter(cl('bat_deficit').is_in([0,-1]))
        .with_columns(rest_inn_runs=cl('fate_runs_ct')+cl('event_runs_ct'))
        .group_by('start_bases_cd','outs_ct','rest_inn_runs')
        .agg(pl.len().alias('n'))
        .sort('outs_ct','start_bases_cd','rest_inn_runs')
        .with_columns(p=cl('n')/(cl('n').sum().over('start_bases_cd','outs_ct')))
        .collect())

# Build up the numpy matrix from this dataframe
m = np.zeros((8,3,15))
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        ind,p = df.filter(cl('start_bases_cd').eq(i),cl('outs_ct').eq(j)).select('rest_inn_runs','p').to_numpy().T
        m[i,j,ind.astype(int)] = p

np.save('tables/p(runs|base,out,need1).npy',m)


''' scratch work, please ignore
'''
#obpd = (df.filter(cl('bat_event_fl').eq('T'))
#          .group_by('game_id','inn_ct','bat_home_id')
#          .agg(cl('on_base').sum())
#          .group_by('on_base')
#          .agg(pl.len())
#          .with_columns(p=cl('len')/cl('len').sum()).sort('on_base'))
#
#obp = df.filter(cl('bat_event_fl').eq('T')).select('on_base').mean().item()
#
#obp_by_pos = (df.filter(cl('bat_event_fl').eq('T'))
#                .group_by('bat_lineup_id')
#                .agg(cl('on_base').mean())
#                .sort('bat_lineup_id')
#                .select('on_base').to_numpy().squeeze())
#
#first_in_inn = (df.filter(cl('inn_new_fl').eq('T'))
#                  .select(cl('bat_lineup_id').value_counts(normalize=True))
#                  .unnest('bat_lineup_id')
#                  .sort('bat_lineup_id')
#                  .select('proportion').to_numpy().squeeze())
#
#prob_on_1b = (df.filter(cl('on_base_so_far')>0,cl('bat_event_fl').eq('T'))
#                .select(~cl('base1_run_id').eq('')).mean().item())
#prob_dp_given_1b = (df.filter(~cl('base1_run_id').eq(''),cl('outs_ct')<2,cl('bat_event_fl').eq('T'))
#                      .select(cl('event_outs_ct').eq(2)).mean().item())
#
#cs = np.zeros(30)
#for _ in range(200000):
#    o = 0; c = 0
#    i = rng.choice(9,p=first_in_inn)
#    while o<3:
#        r = rng.random()<obp_by_pos[i]
#        if c>0 and o<2 and r==0:
#            dpr = rng.random()<0.08012
#            o += dpr
#        o += 1-r
#        c += r
#        i  = (i+1)%9
#    cs[c] += 1
#
#res = shgo(a,bounds=[(0.01,0.15)],workers=-1)
#
#cs/cs.sum()
# 
#p = obpd.select('p').to_numpy().squeeze()
#res = minimize(lambda x: ((nbinom.pmf(np.arange(15),3,1-x)-p)**2).sum(),0.315)
#
#f,ax = plt.subplots()
#ax.plot(p)
#ax.plot(nbinom.pmf(np.arange(15),3,1-obp))
#ax.plot(nbinom.pmf(np.arange(15),3,1-0.308))
#ax.plot(cs/cs.sum(),c='crimson')
#plt.show()
