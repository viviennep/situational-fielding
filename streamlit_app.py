import streamlit as st, numpy as np, polars as pl, duckdb, pathlib
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode, JsCode
cl = pl.col
data_dir = pathlib.Path(__file__).resolve().parent / 'data'

con = duckdb.connect(data_dir / 'leaderboard.duckdb')

last_date = con.query(f"""
        select distinct game_date
        from all_plays
        order by game_date desc
        limit 1;
        """).fetchall()[0][0]

st.set_page_config(layout="wide")

st.markdown(f'''

# Situational Fielding
#### Or, _fielder win probability added over expected (divided by leverage index)_
##### Runner on 1st, tie game, 2 outs, fly ball heading towards the right-centre gap.
- In one world, the right fielder comes crashing in to make the catch, misses, 
and is now laying face down on the grass while the ball rolls to the wall.  
The center fielder picks it up with no dream of a play at the plate, they're losing.
- In another world, the right fielder gets behind the ball, fields it on a hop,
and hits the cutoff man keeping the runner at 3rd and the game tied.

In the eyes of OAA these plays are graded the same: a play can either be a hit or 
an out, and neither was an out, but we know that one fielder hurt his team while 
the other helped. We know this because we can play these little counterfactual 
thought experiments about two fielders in two worlds.  
This is all about a fielding stat which puts a number to those little thought 
experiments. It compares what the player did to all the things he could've done, 
in not just 2 worlds but all the possible worlds, and grades him on how much better 
or worse he did than expected.

_Last updated: {last_date.strftime('%m-%d-%Y')}_

''')

css={'.ag-header-group-cell-label.ag-sticky-label': {'flex-direction': 'column', 'margin': 'auto',
                                                     'font-size': '12pt'}}
st.markdown('''#### Situational Fielding Leaderboard
You can filter columns on mobile by holding down the column header, or on desktop by clicking the menu button when 
you hover over it. This lets you, for example, limit the table to only select center fielders.
''')

@st.cache_data(show_spinner=False)
def load_leaderboard():
    return con.execute('select * from leaderboard order by wpoeli desc').df()

lb = load_leaderboard()
lb['rank'] = None

columnDefs = [
        {'field': 'rank', 
         'headerName': 'Rk', 
         'minWidth': 70, 
         'maxWidth': 70, 
         'filter': False, 
         'sortable': False, 
         'valueGetter': JsCode("function(params) { return params.node.rowIndex + 1; }"),
         'surpressMenu': True},
        {'field': 'fielder_name', 
         'headerName': 'Name', 
         'minWidth': 120, 
         'filter': True, 
         'sortable': False},
        {'field': 'season', 
         'headerName': 'Season', 
         'minWidth':  70, 
         'maxWidth': 70, 
         'filter': True, 
         'sortable': True,},
        {'field': 'team', 
         'headerName': 'Team', 
         'minWidth':  70, 
         'maxWidth': 70, 
         'filter': True, 
         'sortable': True,},
        {'field': 'primary_position', 
         'headerName': 'Pos', 
         'minWidth':  70, 
         'maxWidth': 70, 
         'filter': True, 
         'sortable': True,},
        {'field': 'plays', 
         'headerName': 'Plays', 
         'minWidth':  70, 
         'maxWidth': 70, 
         'filter': True, 
         'sortable': True,},
        {'headerName': "Outs Above Average",
         'headerTooltip': "Outs made compared to expected outs made",
         'children': [
              {'field': 'vioaa', 
               'headerName': 'Outs Above Average (viv)',
               'minWidth': 130,
               'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
               'headerTooltip': "My Outs Above Average",
               'tooltipValueGetter': JsCode("""function(){return "My Outs Above Average"}""")},
              {'field': 'scoaa', 
               'headerName': 'Outs Above Average (statcast)',
               'minWidth': 130,
               'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
               'headerTooltip': "Statcast Outs Above Average",
               'tooltipValueGetter': JsCode("""function(){return "Statcast Outs Above Average"}""")},
         ]},
        {'headerName': "Win Probability",
         'headerTooltip': "Win Probability Added over what's expected based on batted ball",
         'children': [
              {'field': 'wpoe', 
               'headerName': 'Win Prob. over Expected',
               'minWidth': 130,
               'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
               'headerTooltip': "Win Probability Added over Expected",
               'tooltipValueGetter': JsCode("""function(){return "Win Probability Added over Expected"}""")},
              {'field': 'wpoeli', 
               'headerName': 'Situational Fielding',
               'minWidth': 130,
               'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
               'headerTooltip': "WPoE/LI",
               'tooltipValueGetter': JsCode("""function(){return "WPoE/LI"}""")}
         ]}
       ]

gridOptions =  {'defaultColDef': {'flex': 1, 'filterable': True,
								  'groupable': False, 'editable': False, 
                                  'wrapText': True, 'autoHeight': True, 
                                  'suppressMovable': True, 'sortable': True,
                                  'suppressMenu': False},
				'columnDefs': columnDefs,
                'sortModel': [{'colId':'wpoeli','sort':'desc'}], 
				'tooltipShowDelay': 800, 
                'tooltipMouseTrack': True,
                'rowSelection': 'multiple', 
                'rowMultiSelectWithClick': True, 
                'suppressRowDeselection': False, 
                'suppressRowClickSelection': False, 
                'groupSelectsChildren': False, 
                'groupSelectsFiltered': True}

left_col,right_col = st.columns(2)
with left_col.expander('Included Years') :
    years = sorted(lb.season.unique())
    years_select = st.multiselect("Included years", years, years[-1])
with right_col.expander('Included Teams') :
    teams = sorted(lb.team.unique())
    teams_select = st.multiselect("Included Teams", teams, teams, label_visibility='collapsed')

year_filt = lb.season.isin(years_select)
team_filt = lb.team.isin(teams_select)

return_value = AgGrid(lb[year_filt & team_filt], 
       gridOptions=gridOptions,
       update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
       allow_unsafe_jscode=True,
       #fit_columns_on_grid_load=True,
       height=700,
       theme="streamlit",
       key=None,
       custom_css=css)

@st.cache_data(show_spinner=False)
def selected_players_query(sps):
    return con.query(f"""
                       select 
                           resp_fielder_name as name,
                           game_date,
                           des,
                           home_team,
                           away_team,
                           inning,
                           inning_topbot,
                           outs_when_up,
                           base_state,
                           run_diff,
                           out_prob,
                           catch_rate,
                           is_out,
                           li,
                           wp,
                           next_wp,
                           wpa,
                           xwpa,
                           wpoe,
                           wpoeli,
                           play_id
                       from all_plays
                       where is_of_play
                       and (resp_fielder_name, game_year) in (
                       {','.join(f"('{i[0]}',{i[1]})" for i in sps)}
                       );
                 """).df()

st.markdown('''
#### Selected Players Plays
''')
if return_value.selected_rows is None:
    st.write('''Select rows in the table to see individual plays from the selected players''')
else:
    selected_player_seasons = return_value.selected_rows[['fielder_name','season']].to_numpy()
    selected_player_plays = selected_players_query(selected_player_seasons)
    sporty_vid = lambda x: f"https://baseballsavant.mlb.com/sporty-videos?playId={x}"
    selected_player_plays['video_link'] = selected_player_plays['play_id'].apply(sporty_vid)
    selected_player_plays['date'] = selected_player_plays['game_date'].dt.strftime('%Y-%m-%d')
    link_renderer = JsCode("""
    class UrlCellRenderer {
      init(params) {
        this.eGui = document.createElement('a');
        this.eGui.innerText = 'link';          
        this.eGui.href = params.value;        
        this.eGui.target = '_blank';            
        this.eGui.rel = 'noopener';              
        this.eGui.style.textDecoration = 'none'; 
      }
      getGui() {
        return this.eGui;
      }
    }
    """)
    play_columnDefs = [
            {'field': 'name', 
             'headerName': 'Name', 
             'minWidth': 70, 
             'filter': True, 
             'sortable': False, },
            {'field': 'date', 
             'headerName': 'Date', 
             'minWidth':  80, 
             'filter': True, 
             'sortable': True,},
            {'field': 'des', 
             'headerName': 'Description', 
             'minWidth': 150, 
             'wrapText': False, 
             'tooltipField': 'des',
             'filter': False, 
             'sortable': False,},
            {'headerName': "Game State",
             'headerTooltip': "State of the game at the beginning of the play",
             'children': [
                {'field': 'home_team', 
                 'headerName': 'Home', 
                 'minWidth': 50, 
                 'filter': False, 
                 'sortable': False,},
                {'field': 'away_team', 
                 'headerName': 'Away', 
                 'minWidth': 50, 
                 'filter': False, 
                 'sortable': False,},
                {'field': 'inning', 
                 'headerName': 'Inn', 
                 'minWidth': 40, 
                 'maxWidth': 40, 
                 'filter': False, 
                 'sortable': True,},
                {'field': 'inning_topbot', 
                 'headerName': 'Half', 
                 'minWidth': 40, 
                 'maxWidth': 40, 
                 'filter': False, 
                 'sortable': False,},
                {'field': 'outs_when_up', 
                 'headerName': 'Outs', 
                 'minWidth': 40, 
                 'maxWidth': 50, 
                 'filter': False, 
                 'sortable': False,},
                {'field': 'base_state', 
                 'headerName': 'Base State', 
                 'minWidth': 40, 
                 'maxWidth': 50, 
                 'filter': False, 
                 'sortable': False,},
                {'field': 'run_diff', 
                 'headerName': 'Home Team Run Diff', 
                 'minWidth':  70, 
                 'maxWidth': 70, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 0,
                 'filter': False, 
                 'sortable': False,},
            ]},
            {'headerName': "Out Probability",
             'headerTooltip': "Catch Probabilities on the Play",
             'children': [
                {'field': 'out_prob', 
                 'headerName': 'Catch Prob (viv)', 
                 'minWidth':  70, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                 'filter': True, 
                 'sortable': True,},
                {'field': 'catch_rate', 
                 'headerName': 'Catch Prob (sc)', 
                 'minWidth':  70, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                 'filter': True, 
                 'sortable': True,},
                {'field': 'is_out', 
                 'headerName': 'Out Made', 
                 'minWidth':  70, 
                 'maxWidth': 70, 
                 'filter': True, 
                 'sortable': True,},
            ]},
            {'headerName': "Win Probability",
             'headerTooltip': "Catch Probabilities on the Play",
             'children': [
                # just for debugging
                #{'field': 'wp', 
                # 'headerName': 'Win Probability', 
                # 'minWidth':  95, 
                # 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                # 'filter': True, 
                # 'sortable': True,},
                #{'field': 'next_wp', 
                # 'headerName': 'Next Win Probability', 
                # 'minWidth':  95, 
                # 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                # 'filter': True, 
                # 'sortable': True,},
                {'field': 'wpa', 
                 'headerName': 'Win Probability Added', 
                 'minWidth':  95, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                 'filter': True, 
                 'sortable': True,},
                {'field': 'li', 
                 'headerName': 'Leverage Index', 
                 'minWidth':  90, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                 'filter': True, 
                 'sortable': True,},
                {'field': 'xwpa', 
                 'headerName': 'Expected WPA', 
                 'minWidth':  90, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                 'filter': True, 
                 'sortable': True,},
                {'field': 'wpoe', 
                 'headerName': 'Win Probability over Expected', 
                 'minWidth':  95, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                 'filter': True, 
                 'sortable': True,},
                {'field': 'wpoeli', 
                 'headerName': 'WPoE/LI', 
                 'minWidth':  85, 
                 'type' : ['numericColumn', 'customNumericFormat'], 'precision': 2,
                 'filter': True, 
                 'sortable': True,},
            ]},
            {'field': 'video_link', 
             'headerName': 'Video', 
             'minWidth':  50, 
             'cellRenderer': link_renderer,
             'filter': False, 
             'sortable': False,},
        ]
    play_gridOptions =  {'defaultColDef': {'flex': 1, 'filterable': True,
                                      'groupable': False, 'editable': False, 
                                      'sortable': True,
                                      'wrapText': True, 'autoHeight': True, 
                                      'wrapHeaderText': True,
                                      'autoHeaderHeight': True,
                                      'suppressMovable': True,
                                      'suppressMenu': False},
                    'columnDefs': play_columnDefs,
                    'sortModel': [{'colId':'game_date','sort':'desc'}], 
                    'tooltipShowDelay': 800, 
                    'tooltipMouseTrack': True,
                    'rowSelection': 'multiple', 
                    'rowMultiSelectWithClick': True, 
                    'initialState': {'rowSelection': [0,1]},
                    'suppressRowDeselection': False, 
                    'suppressRowClickSelection': False, 
                    'groupSelectsChildren': False, 
                    'groupSelectsFiltered': True}
    plays_grid = AgGrid(selected_player_plays, 
           gridOptions=play_gridOptions,
           update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.VALUE_CHANGED,
           allow_unsafe_jscode=True,
           #fit_columns_on_grid_load=True,
           height=700,
           theme="streamlit",
           key=None,
           custom_css=css)


st.markdown('''#### Calculation Details''')
wpoe_exp = st.expander('Expected Win Probability ')
with wpoe_exp:
	st.markdown(r'''
##### Win Probability over Expected
The idea is simple, how much better or worse was the win probability after a play $\mathrm{WP}_f$ 
compared to what the win probability was expected to be based on how the ball was hit. 
$$
\mathrm{WPoE} = \mathrm{WP}_f - \mathrm{E}[\mathrm{WP}_f\ |\ \mathrm{game\ state,\ BIP\ Characteristics}]
$$
This is conceptually the same framework as Outs Above Average, which says how much better or worse
the real number of outs (1 or 0) was than the expected number of outs (catch probability) after the
ball is put in play.  
$\mathrm{WPoE}$ takes this one step further. Rather than only working in terms of outs, 
this looks at the result of the play in terms of the whole game state: inning, outs, baserunners,
and run differential.

###### Factoring the model

The tricky part is determining what we expect the next win probability to be after a ball is put in 
play, because to do this we need to assign a probability to every possible next game state. This is a 
pretty unwieldy problem, but it can be broken down into pieces.  

First we can rewrite the expectation:
$$
\mathrm{E}[\mathrm{WP}_f\ |\ \mathrm{GS,\ BIP}] = \sum_{o\in\mathrm{outcomes}}{p(o\ |\ \mathrm{BIP})
\mathrm{E}[\mathrm{WP}_f\ |\ o\mathrm{,\ GS}]}
$$
where the outcomes, $o$, can be things like out, single, double, etc. We _could_ make some machine 
learning model which returns these probabilities from the ball-in-play ($\mathrm{BIP}$) characteristics 
(exit velocity, launch angle, azimuthal angle), along with all the normal catch probability features 
(hang time, distance to cover, wall features), but for the sake of accuracy its advantageous to divide
the problem into 2 branches: out or no out.
$$
\mathrm{E}[\mathrm{WP}_f\ |\ \mathrm{GS,\ BIP}] = p(\mathrm{out}\ |\ \mathrm{distance,\ hang\ time,\ wall})
\mathrm{E}[\mathrm{WP}_f\ |\ \mathrm{GS,\ out}] + p(\mathrm{no\ out})
\mathrm{E}[\mathrm{WP}_f\ |\ \mathrm{GS,\ no\ out}] 
$$
Doing this introduces catch probability into the metric, 
$p(\mathrm{catch}\ |\ \mathrm{distance,\ hang\ time,\ wall})$ and allows the remaing propabilities 
to become conditional on it,
$$
p(o) = p(o\ |\ \mathrm{no\ out})p(\mathrm{no\ out}) = p(o\ |\ \mathrm{no\ out})\left(1-p(\mathrm{out})\right)
$$
These probabilities can be determined from a machine learning model—like xwOBA—which we can expect to 
be rather accurate now that its scope has been reduced to only non-out balls in play.

###### Expected win probabilities

Having factored the problem with our catch probability & hit outcome probability models, the remaining
bite-sized expectations are all in this form:
$$
\mathrm{E}[\mathrm{WP}_f\ |\ \mathrm{GS,\ outcome,\ BIP}]
$$ 
this involves considering every possible next game state provided the BIP outcome and BIP charcteristics.
In English this means considerring the situation, e.g. runners on 1st and 2nd, low line drive single to 
right field, and determining what's the probability of: the bases clearing 2 runs score, 1st and 3rd 1 run,
bases loaded no runs, 1 run scoring but one runner being thrown out at home, etc. etc. All the possibile
next base states. The vast majority of the complexity in these game state transitions comes from 
baserunner movement. To address this I opted to model each runner's advancement individually, placing a 
probability distribution the possible bases he could advance (0, 1, 2, 3 or out), and wrap these 
advancements in some transition logic which determines the final state of the bases provided each 
individual runner's advancement. I'm more than certain there will be holes in this logic, but it's 
certainly good enough for play valuation.  
With the probability of each next base/out/run state determined from this transition mapper & baserunner 
advancement probabilities, the expected win probability of the next state can be determined easily
$$
\mathrm{E}[\mathrm{WP}_f] = \sum_{gs\in\mathrm{next\ game\ state}}{p(gs)\mathrm{WP}_{gs}}
$$

''')

st.markdown('''
### List of issues
- Extra innings can be weird. I have fixed 2 issues with extras already but some weird plays remain.  
PCA's most valuable play at the moment is a 99% routine pop up in the 11th where the xWPA is -18%. The
rest of his most valuable plays look normal, but this is an obvious outlier.  
Similarly, Fernando Tatis Jr gets 3 huge plays for catching routine pop outs in bot 9 and bot 10.
- Sac flies can be pretty harsh to fielders. The baserunner sub model doesn't account for who the runner is,
it just assumes they're average, so there are cases where a fast runner is on 3rd and the fielder really has 
no shot where the model dings the fielder unfairly.
- I've noticed sometimes when a play challenged the WPA is incorrectly 0, which I hypothesize is because the 
"next event" in the game log was the challenge event and not the actual next play.
- Naturally this includes the _whole_ play, and not just what the fielder can control. Rafaela has
a play where he misses a ball, commits a throwing error, but then Marcelo Mayer gets the ball
and _also_ commits a throwing error. Ceddanne is considered the responsible fielder for the
whole play. Tough look for him but splitting up plays into sub-events is beyond the scope of this
little website.
''')

