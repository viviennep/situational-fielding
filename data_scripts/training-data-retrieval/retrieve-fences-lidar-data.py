import numpy as np, requests, json, polars as pl, matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from scipy.interpolate import interp1d
from io import StringIO
cl = pl.col

headers = {'User-Agent': 'Mozilla/5.0'}

url  = f"https://builds.mlbstatic.com/baseballsavant.mlb.com/v1/sections/player-update/builds/"
url += "24bd00d6b76619c30d68da4cd52d49d32e141f1e/scripts/build/index.bundle.js"

res = requests.get(url, headers=headers).content.decode('utf-8')
start = res.index('year_min,year_max,venue_id')
end = res[start:].index('"')
fences_csv = res[start:start+end].replace('\\n','\n')
fences_df = pl.read_csv(StringIO(fences_csv))

team_to_park_map = {'Miami Marlins'        : 'loanDepot park',
                    'Baltimore Orioles'    : 'Oriole Park at Camden Yards',
                    'Chicago Cubs'         : 'Wrigley Field',
                    'New York Yankees'     : 'Yankee Stadium',
                    'Cleveland Guardians'  : 'Progressive Field',
                    'Cincinnati Reds'      : 'Great American Ball Park',
                    'Milwaukee Brewers'    : 'American Family Field',
                    'Arizona Diamondbacks' : 'Chase Field',
                    'Atlanta Braves'       : 'Truist Park',
                    'Houston Astros'       : 'Minute Maid Park',
                    'Chicago White Sox'    : 'Rate Field',
                    'Kansas City Royals'   : 'Kauffman Stadium',
                    'Texas Rangers'        : 'Globe Life Field',
                    'Washington Nationals' : 'Nationals Park',
                    'Oakland Athletics'    : 'Oakland Coliseum',
                    'Athletics'            : 'Sutter Health Park',
                    'Seattle Mariners'     : 'T-Mobile Park',
                    'St. Louis Cardinals'  : 'Busch Stadium',
                    'Boston Red Sox'       : 'Fenway Park',
                    'Detroit Tigers'       : 'Comerica Park',
                    'San Diego Padres'     : 'Petco Park',
                    'Los Angeles Dodgers'  : 'Dodger Stadium',
                    'San Francisco Giants' : 'Oracle Park',
                    'New York Mets'        : 'Citi Field',
                    'Cleveland Indians'    : 'Progressive Field',
                    'Los Angeles Angels'   : 'Angel Stadium',
                    'Pittsburgh Pirates'   : 'PNC Park',
                    'Tampa Bay Rays'       : 'Tropicana Field',
                    'Toronto Blue Jays'    : 'Rogers Centre',
                    'Philadelphia Phillies': 'Citizens Bank Park',
                    'Colorado Rockies'     : 'Coors Field',
                    'Minnesota Twins'      : 'Target Field'}

team_to_sc_abbr  = {'Miami Marlins'        : 'MIA',
                    'Baltimore Orioles'    : 'BAL',
                    'Chicago Cubs'         : 'CHC',
                    'New York Yankees'     : 'NYY',
                    'Cleveland Guardians'  : 'CLE',
                    'Cincinnati Reds'      : 'CIN',
                    'Milwaukee Brewers'    : 'MIL',
                    'Arizona Diamondbacks' : 'AZ',
                    'Atlanta Braves'       : 'ATL',
                    'Houston Astros'       : 'HOU',
                    'Chicago White Sox'    : 'CWS',
                    'Kansas City Royals'   : 'KC',
                    'Texas Rangers'        : 'TEX',
                    'Washington Nationals' : 'WSH',
                    'Oakland Athletics'    : 'OAK',
                    'Athletics'            : 'ATH',
                    'Seattle Mariners'     : 'SEA',
                    'St. Louis Cardinals'  : 'STL',
                    'Boston Red Sox'       : 'BOS',
                    'Detroit Tigers'       : 'DET',
                    'San Diego Padres'     : 'SD',
                    'Los Angeles Dodgers'  : 'LAD',
                    'San Francisco Giants' : 'SF',
                    'New York Mets'        : 'NYM',
                    'Cleveland Indians'    : 'CLE',
                    'Los Angeles Angels'   : 'LAA',
                    'Pittsburgh Pirates'   : 'PIT',
                    'Tampa Bay Rays'       : 'TB',
                    'Toronto Blue Jays'    : 'TOR',
                    'Philadelphia Phillies': 'PHI',
                    'Colorado Rockies'     : 'COL',
                    'Minnesota Twins'      : 'MIN'}


# We can just say that the rays play in new york, 
# but for the A's we need to grab the outfield from google earth
latlongs = np.array([[-121.5129217592458,38.58018689232059],
                     [-121.5129175993659,38.58019799595177],
                     [-121.5129117073863,38.58021456609472],
                     [-121.5129032810098,38.58023667715908],
                     [-121.5128974430451,38.58025341130961],
                     [-121.5128901612496,38.58027261979277],
                     [-121.5128808714945,38.58029317182508],
                     [-121.5128720426001,38.58031524228527],
                     [-121.5128634452182,38.58033847926828],
                     [-121.5128556766013,38.58036194491245],
                     [-121.5128452340928,38.58038771291763],
                     [-121.5128322830106,38.58041977387293],
                     [-121.5128267107691,38.58043571041105],
                     [-121.5128206170598,38.58045144561893],
                     [-121.5128154678136,38.58046549451220],
                     [-121.5128171380916,38.58047172523028],
                     [-121.5128244113363,38.58049425437407],
                     [-121.5128316414655,38.58051663983900],
                     [-121.5128407407706,38.58054456176210],
                     [-121.5128493203810,38.58057005822619],
                     [-121.5128557650724,38.58058804868565],
                     [-121.5128636382132,38.58061067873021],
                     [-121.5128708447481,38.58063270403446],
                     [-121.5128772874124,38.58065326593819],
                     [-121.5128835560080,38.58067196903490],
                     [-121.5128912203144,38.58069836308935],
                     [-121.5128967451135,38.58071680561391],
                     [-121.5129017015101,38.58073274787144],
                     [-121.5129092628667,38.58075802373465],
                     [-121.5129147255591,38.58077660016633],
                     [-121.5129196618336,38.58079278153256],
                     [-121.5129247784924,38.58080949405876],
                     [-121.5129312100120,38.58083063281745],
                     [-121.5129383258488,38.58085383829267],
                     [-121.5129433700146,38.58087089255893],
                     [-121.5129481689724,38.58088612009702],
                     [-121.5129508962575,38.58089172136964],
                     [-121.5129695937787,38.58090745685592],
                     [-121.5129824493571,38.58091634991894],
                     [-121.5129915763997,38.58092525746674],
                     [-121.5130066920948,38.58093748728137],
                     [-121.5130342439824,38.58096192548269],
                     [-121.5130811913189,38.58099906263975],
                     [-121.5130991932995,38.58101339361801],
                     [-121.5131140304106,38.58102647388836],
                     [-121.5131276230185,38.58103728124140],
                     [-121.5131353124302,38.58104218421974],
                     [-121.5131464190645,38.58104894069008],
                     [-121.5131687347290,38.58105941524884],
                     [-121.5131991708540,38.58107417344399],
                     [-121.5132439769880,38.58109250665827],
                     [-121.5132598314792,38.58109897651325],
                     [-121.5132921774679,38.58110880429657],
                     [-121.5133461368386,38.58112347059930],
                     [-121.5133686098050,38.58112943008066],
                     [-121.5134029645484,38.58113850810172],
                     [-121.5134332943884,38.58114443134232],
                     [-121.5134684021886,38.58114972073478],
                     [-121.5134965912610,38.58115430966004],
                     [-121.5135227611943,38.58115766502374],
                     [-121.5135473005810,38.58116090100589],
                     [-121.5135778007251,38.58116360539685],
                     [-121.5136040444012,38.58116533457701],
                     [-121.5136467292578,38.58116626669991],
                     [-121.5136823436103,38.58116644494496],
                     [-121.5137214930462,38.58116579156964],
                     [-121.5137600309852,38.58116395868302],
                     [-121.5137984015594,38.58116092154729],
                     [-121.5138309554564,38.58115747713269],
                     [-121.5138611141983,38.58115309882054],
                     [-121.5139023875381,38.58114611645447],
                     [-121.5139448707495,38.58113711950534],
                     [-121.5139993615135,38.58112346595287],
                     [-121.5140329192082,38.58111424798770],
                     [-121.5140734342934,38.58021046546152]])

# To convert lat/long deltas to ft:
# 1 deg lat = 60 nautical miles = 364567 ft
# 1 deg lon = 364567*cos(lat)
deg_to_ft = 364567
long_factor = np.cos(latlongs[:,1]*np.pi/180)

# find origin (home plate) point and centre there
ori_ind = np.argmax([i[i>0].min() for i in squareform(pdist(latlongs))])
latlongs -= latlongs[ori_ind]

latlongs[:,0] *= deg_to_ft*long_factor
latlongs[:,1] *= deg_to_ft

# get home plate out of there
latlongs = latlongs[np.arange(len(latlongs))!=ori_ind]

# convert to a polar representation so we can evaluate wall distance
# at each integer degree from -45 to 45, which is how the fences lidar works
theta = np.arctan2(*latlongs.T[::-1])
radii = np.linalg.norm(latlongs,axis=-1)
order = np.argsort(theta)
theta = theta[order]
radii = radii[order]

r = interp1d((theta-theta.min())*180/np.pi-45,radii,kind='linear',fill_value='extrapolate')

angles = np.arange(-45,46)
r_ang  = r(angles)
xy_ang = np.zeros((2,len(angles)))
xy_ang[0] = r_ang*np.sin(angles*np.pi/180)
xy_ang[1] = r_ang*np.cos(angles*np.pi/180)

# Build dict to turn into dataframe
park_dict = {'year_min': 2025,
             'year_max': 2025,
             'venue_id': 0,    # idk, not using this
             'venue_name_short': 'Sutter Health Park',
             'park': 'Sutter Health Park',
             'angle': angles,
             'fence_height': fences_df.select('fence_height').median().item(), # don't know wall heights
             'fence_distance': r_ang,
             'fence_x': xy_ang[0],
             'fence_y': xy_ang[1],
             'fence_x_inches': 12*xy_ang[0],
             'fence_y_inches': 12*xy_ang[1]}

fences_df = pl.concat([fences_df,pl.DataFrame(park_dict).cast(dict(fences_df.schema))])

park_to_team_map = {v:k for k,v in team_to_park_map.items()}

fences_df = (fences_df.with_columns(team=cl('park').replace(park_to_team_map))
                      .with_columns(team_abbr=cl('team').replace(team_to_sc_abbr)))
fences_df.write_parquet('../../data/fences-lidar.parquet')

