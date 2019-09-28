from myScraper import parse_season

FRANCHISES = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal',
              'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan', 'mia', 'min',
              'nwe', 'nor', 'nyg', 'nyj', 'rai', 'phi', 'pit', 'sdg', 'sfo',
              'sea', 'ram', 'tam', 'oti', 'was']

FRANCHISE_NAMES = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens',
                   'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears',
                   'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys',
                   'Denver Broncos', 'Detroit Lions', 'Green Bay Packers',
                   'Houston Texans', 'Indianapolis Colts',
                   'Jacksonville Jaguars', 'Kansas City Chiefs',
                   'Miami Dolphins', 'Minnesota Vikings',
                   'New England Patriots', 'New Orleans Saints',
                   'New York Giants', 'New York Jets', 'Oakland Raiders',
                   'Philadelphia Eagles', 'Pittsburgh Steelers',
                   'San Diego Chargers', 'San Francisco 49ers',
                   'Seattle Seahawks', 'St. Louis Rams', 'Tampa Bay Buccaneers',
                   'Tennessee Titans', 'Washington Redskins']

COL_NAMES = ['year', 'team', 'verbose_name', 'week', 'day', 'date', 'boxscore_url',
             'boxscore_text', 'result', 'OT', 'record',
             'at', 'opponent', 'team_score', 'opp_score', 'off_first_downs',
             'off_total_yds', 'off_pass_yds', 'off_rush_yds',
             'off_turn_overs', 'def_first_downs', 'def_total_yds',
             'def_pass_yds', 'def_rush_yds', 'def_turn_overs', 'off_exp_pts',
             'def_exp_pts', 'specialTm_exp_pts', 'Won_Toss', 'Roof', 'Surface',
             'Vegas_Line_Close', 'O/U_Line', 'O/U_Result', 'aFirst_Downs', 'hFirst_Downs',
             'aRush-Yds-Tds', 'hRush-Yds-Tds', 'aCmp-Att-Yd-TD-INT', 'hCmp-Att-Yd-TD-INT',
             'aSacked-Yds', 'hSacked-Yds', 'aNet_Pass_Yds', 'hNet_Pass_Yds',
             'aTotal_Yds', 'h_Total_Yds', 'aFumbles-Lost', 'hFumbles-Lost',
             'aTurnovers', 'hTurnovers', 'aPenalties-Yds', 'h-Penalties-Yds',
             'aThird_Down_Conv', 'hThird_Down_Conv', 'aFourth_Down_Conv', 'hFourth_Down_Conv',
             'aTime_of_Possesion', 'hTime_of_Possesion']

Franchise_Dict = dict(zip(FRANCHISES, FRANCHISE_NAMES))

start_year = 2009
end_year = 2018

def build_master():
    filename = 'nfl_master_%s-%s.csv' % (start_year, end_year)
    with open(filename, 'w') as f:
        f.write(','.join(COL_NAMES) + '\n')
        for team, verbose_name in Franchise_Dict.items():
            for year in range(start_year, end_year + 1):
                try:
                    for season in parse_season(team, verbose_name, year):
                        print('{} - {}'.format(verbose_name, year))
                        f.write(season + '\n')
                except Exception as e:
                    print(e)
                    pass

if __name__ == '__main__':
    build_master()