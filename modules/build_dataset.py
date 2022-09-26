import errno
from shutil import copyfile
import argparse
import sys
from datetime import datetime
import os
from pfr_scraper import parse_season


FRANCHISES = [
    'crd',
    'atl',
    'rav',
    'buf',
    'car',
    'chi',
    'cin',
    'cle',
    'dal',
    'den',
    'det',
    'gnb',
    'htx',
    'clt',
    'jax',
    'kan',
    'mia',
    'min',
    'nwe',
    'nor',
    'nyg',
    'nyj',
    'rai',
    'phi',
    'pit',
    'sdg',
    'sfo',
    'sea',
    'ram',
    'tam',
    'oti',
    'was',
]

FRANCHISE_NAMES = [
    'Arizona Cardinals',
    'Atlanta Falcons',
    'Baltimore Ravens',
    'Buffalo Bills',
    'Carolina Panthers',
    'Chicago Bears',
    'Cincinnati Bengals',
    'Cleveland Browns',
    'Dallas Cowboys',
    'Denver Broncos',
    'Detroit Lions',
    'Green Bay Packers',
    'Houston Texans',
    'Indianapolis Colts',
    'Jacksonville Jaguars',
    'Kansas City Chiefs',
    'Miami Dolphins',
    'Minnesota Vikings',
    'New England Patriots',
    'New Orleans Saints',
    'New York Giants',
    'New York Jets',
    'Oakland Raiders',
    'Philadelphia Eagles',
    'Pittsburgh Steelers',
    'San Diego Chargers',
    'San Francisco 49ers',
    'Seattle Seahawks',
    'St. Louis Rams',
    'Tampa Bay Buccaneers',
    'Tennessee Titans',
    'Washington Redskins',
]

COL_NAMES = [
    'year',
    'team',
    'verbose_name',
    'week',
    'day',
    'boxscore_url',
    'time',
    'boxscore_text',
    'result',
    'OT',
    'record',
    'at',
    'opponent',
    'team_score',
    'opp_score',
    'off_first_downs',
    'off_total_yds',
    'off_pass_yds',
    'off_rush_yds',
    'off_turn_overs',
    'def_first_downs',
    'def_total_yds',
    'def_pass_yds',
    'def_rush_yds',
    'def_turn_overs',
    'off_exp_pts',
    'def_exp_pts',
    'specialTm_exp_pts',
    'Won_Toss',
    'Roof',
    'Surface',
    'Vegas_Line_Close',
    'O/U_Line',
    'O/U_Result',
    'aFirst_Downs',
    'hFirst_Downs',
    'aRush-Yds-Tds',
    'hRush-Yds-Tds',
    'aCmp-Att-Yd-TD-INT',
    'hCmp-Att-Yd-TD-INT',
    'aSacked-Yds',
    'hSacked-Yds',
    'aNet_Pass_Yds',
    'hNet_Pass_Yds',
    'aTotal_Yds',
    'h_Total_Yds',
    'aFumbles-Lost',
    'hFumbles-Lost',
    'aTurnovers',
    'hTurnovers',
    'aPenalties-Yds',
    'h-Penalties-Yds',
    'aThird_Down_Conv',
    'hThird_Down_Conv',
    'aFourth_Down_Conv',
    'hFourth_Down_Conv',
    'aTime_of_Possesion',
    'hTime_of_Possesion',
    'hOff_snap_count',
    'hDef_snap_count',
    'hST_snap_count',
    'aOff_snap_count',
    'aDef_snap_count',
    'aST_snap_count',
]

FRANCHISE_DICT = dict(zip(FRANCHISES, FRANCHISE_NAMES))


def get_filename(start_year, end_year):
    return (f'nfl_master_{start_year}-{end_year}.csv')


def backup_existing_master(filename):
    old_file, new_file = filename, f'backup_{filename}'
    if not os.path.exists(old_file):
        print(f'{old_file} does not exist yet')
        return
    print(f'Backing up\nFrom: {old_file}\nTo: {new_file}')
    copyfile(old_file, new_file)


def restore_backup(filename):
    old_file, new_file = f'backup_{filename}', filename
    if not os.path.exists(old_file):
        print(f'{old_file} does not exist yet')
        return
    print(f'Restoring\nFrom: {old_file}\nTo: {new_file}')
    copyfile(old_file, new_file)


def parse(file_descriptor, team, verbose_name, year, week=None):
    for season in parse_season(team, verbose_name, year, week):
        print(f'{verbose_name} - {year}')
        file_descriptor.write(season + '\n')


# recreate master spreadsheet
def build_master(start_year, end_year):
    filename = get_filename(start_year, end_year)

    backup_existing_master(filename)

    with open(filename, 'w') as file_descriptor:
        file_descriptor.write(','.join(COL_NAMES) + '\n')
        for team, verbose_name in FRANCHISE_DICT.items():
            for year in range(start_year, end_year + 1):
                try:
                    parse(file_descriptor, team, verbose_name, year)
                except Exception as err:
                    restore_backup(filename)
                    raise err


# appends new week to master spreadsheet
def add_new_week(year, week):
    filename = get_filename(START_YEAR, END_YEAR)

    backup_existing_master(filename)

    with open(filename, 'a') as file_descriptor:
        for team, verbose_name in FRANCHISE_DICT.items():
            try:
                parse(file_descriptor, team, verbose_name, year, week)
            except Exception as err:
                restore_backup(filename)
                raise err


if __name__ == '__main__':
    PARSED = argparse.ArgumentParser()
    PARSED.add_argument('-s', '--startYear', type=int, default=2009)
    PARSED.add_argument('-e', '--endYear', type=int, default=datetime.now().year)
    PARSED.add_argument('-w', '--week', type=int)
    PARSED.add_argument('-rb', '--rebuild', type=bool, default=False)
    PARSED.add_argument('-rs', '--restore', type=bool, default=False)
    ARGS = PARSED.parse_args()
    START_YEAR, END_YEAR, WEEK, REBUILD, RESTORE = (
        ARGS.startYear,
        ARGS.endYear,
        ARGS.week,
        ARGS.rebuild,
        ARGS.restore,
    )

    if RESTORE:
        restore_backup(get_filename(START_YEAR, END_YEAR))
    elif REBUILD:
        try:
            build_master(START_YEAR, END_YEAR)
        except KeyboardInterrupt:
            print('Interrupted')
            restore_backup(get_filename(START_YEAR, END_YEAR))
    else:
        if WEEK is None:
            print('Must specify an end week if not rebuilding/restoring master')
            sys.exit(errno.EINVAL)
        add_new_week(END_YEAR, WEEK)

