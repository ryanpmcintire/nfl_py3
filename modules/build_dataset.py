import errno
from shutil import copyfile
from myScraper import parse_season
import argparse
import sys
from datetime import datetime

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

COL_NAMES = ['year', 'team', 'verbose_name', 'week', 'day', 'boxscore_url', 'time',
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


def get_filename(start_year, end_year):
    return f'nfl_master_{start_year}-{end_year}.csv'


def backup_existing_master(filename):
    oldFile, newFile = filename, 'backup_' + filename
    print(f'Backing up\nFrom: {oldFile}\nTo: {newFile}')
    copyfile(oldFile, newFile)


def restore_backup(filename):
    oldFile, newFile = 'backup_' + filename, filename
    print(f'Restoring\nFrom: {oldFile}\nTo: {newFile}')
    copyfile(oldFile, newFile)


def parse(f, team, verbose_name, year, week=None):
    for season in parse_season(team, verbose_name, year, week):
        print(f'{verbose_name} - {year}')
        f.write(season + '\n')


# recreate master spreadsheet
def build_master(start_year, end_year):
    filename = get_filename(start_year, end_year)

    backup_existing_master(filename)

    with open(filename, 'w') as f:
        f.write(','.join(COL_NAMES) + '\n')
        for team, verbose_name in Franchise_Dict.items():
            for year in range(start_year, end_year + 1):
                try:
                    parse(f, team, verbose_name, year)
                except Exception as e:
                    raise(e)


# appends new week to master spreadsheet
def add_new_week(year, week):
    filename = get_filename(start_year, end_year)

    backup_existing_master(filename)

    with open(filename, 'a') as f:
        for team, verbose_name in Franchise_Dict.items():
            try:
                parse(f, team, verbose_name, year, week)
            except Exception as e:
                raise(e)


if __name__ == '__main__':
    parsed = argparse.ArgumentParser()
    parsed.add_argument('-s', '--startYear', type=int, default=2009)
    parsed.add_argument('-e', '--endYear',
                        type=int, default=datetime.now().year)
    parsed.add_argument('-w', '--week', type=int)
    parsed.add_argument('-rb', '--rebuild', type=bool, default=False)
    parsed.add_argument('-rs', '--restore', type=bool, default=False)
    args = parsed.parse_args()
    start_year, end_year, week, rebuild, restore = args.startYear, args.endYear, args.week, args.rebuild, args.restore

    if (restore):
        restore_backup(get_filename(start_year, end_year))
    elif (rebuild):
        build_master(start_year, end_year)
    else:
        if week is None:
            print('Must specify an end week if not rebuilding/restoring master')
            sys.exit(errno.EINVAL)
        add_new_week(end_year, week)
