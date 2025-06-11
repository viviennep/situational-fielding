import numpy as np, polars as pl
cl = pl.col

def calc_dist2wall(player_pos,wall_pos,directions,wall_h):
    directions = directions/np.linalg.norm(directions,axis=-1)[:,None] # just making sure
    A,C = directions.T
    B,D = (wall_pos[:-1] - wall_pos[1:]).T
    E,F = (wall_pos[:-1] - player_pos[:,None]).swapaxes(0,2).swapaxes(1,2)
    Delta = np.einsum('i,j->ij',A,D) - np.einsum('i,j->ij',C,B)
    t = (E*D - B*F)/Delta
    u = (np.einsum('i,ij->ij',A,F) - np.einsum('i,ij->ij',C,E))/Delta
    mask = (t>0) & ((u>0) & (u<=1))
    multis = np.where(mask.sum(1)>1)[0]
    for multi in multis:
        new = np.full(t.shape[1],False)
        closest = np.where(mask[multi])[0][t[multi][mask[multi]].argmin()]
        new[closest] = True
        mask[multi] = new
    ds = -np.ones(len(t))
    heights = ds.copy()
    ds[mask.any(1)] = t[mask]
    intersections = player_pos + ds[:,None]*directions
    heights[mask.any(1)] = ((wall_h[1:] + wall_h[:-1])/2)[np.where(mask)[1]]
    return ds,heights,intersections

def calc_wall_properties(ofp,fences_df,verbose=False):
    start_pos      = ofp.select('start_pos_x','start_pos_y').to_numpy()
    land_pos       = ofp.select('landing_pos_x','landing_pos_y').to_numpy()
    diff_vec       = land_pos-start_pos
    dists          = np.linalg.norm(diff_vec,axis=1)
    diff_vec_hat   = diff_vec/dists[:,None]
    start_pos_hat  = start_pos/np.linalg.norm(start_pos,axis=1)[:,None]
    dots           = np.einsum('ij,ij->i',diff_vec_hat,start_pos_hat)
    dets           = np.einsum('ij,ij,j->i',diff_vec_hat,start_pos_hat[:,::-1],np.array((1,-1)))
    angles         = 180-np.arctan2(dets,dots)*180/np.pi
    teams          = ofp.select('home_team').to_numpy().squeeze()
    team_l,team_i  = np.unique(teams,return_inverse=True)
    wall_rad_dist  = np.zeros_like(dists)
    wall_dir_dist  = np.zeros_like(dists)
    wall_dir_h     = np.zeros_like(dists)
    wall_ball_dist = np.zeros_like(dists)
    wall_min_dist  = np.zeros_like(dists)
    for i,team in enumerate(team_l):
        if verbose: print(team)
        park_segs = fences_df.filter(cl('team_abbr').eq(team)).select('fence_x','fence_y').to_numpy()
        park_hs   = fences_df.filter(cl('team_abbr').eq(team)).select('fence_height').to_numpy().squeeze()
        park_mask = team_i == i
        ds,heights,intersections  = calc_dist2wall(start_pos[park_mask], park_segs, start_pos_hat[park_mask], park_hs)
        wall_rad_dist[park_mask]  = ds
        ds,heights,intersections  = calc_dist2wall(start_pos[park_mask], park_segs, diff_vec_hat[park_mask], park_hs)
        wall_dir_dist[park_mask]  = ds
        wall_dir_h[park_mask]     = heights
        ds,heights,intersections  = calc_dist2wall(land_pos[park_mask], park_segs, diff_vec_hat[park_mask], park_hs)
        wall_ball_dist[park_mask] = ds
        wall_min_dist[park_mask]  = np.linalg.norm((land_pos[park_mask,None] - park_segs),axis=-1).min(-1)
    wall_dict = {'play_id': ofp.select('play_id').to_numpy().squeeze(),
                 'angle': angles, 
                 'dist': dists, 
                 'wall_dist_start': wall_rad_dist,
                 'wall_dist_land': wall_ball_dist,
                 'wall_dist_ball_dir': wall_dir_dist,
                 'wall_min_dist': wall_min_dist,
                 'wall_height': wall_dir_h}
    return wall_dict

