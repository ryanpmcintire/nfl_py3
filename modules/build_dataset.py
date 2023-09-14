import errno
from shutil import copyfile
import argparse
import sys
from datetime import datetime
import os
from pfr_scraper import parse_season
import config

COL_NAMES = [
    'year',
    'team',
    'verbose_name',
    'week',
    'day',
    'boxscore_url',
    'time',
    'boxscore_text',
    *config.SHARED_COLS,
]

FRANCHISE_DICT = dict(zip(config.FRANCHISE_ABBRV, config.FRANCHISES))


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
def build_master(start_year, end_year, end_week=None):
    filename = get_filename(start_year, end_year)

    backup_existing_master(filename)
    # when not ending at week do entire season
    if not end_week:
        end_year += 1
    with open(filename, 'w') as file_descriptor:
        file_descriptor.write(','.join(COL_NAMES) + '\n')
        for team, verbose_name in FRANCHISE_DICT.items():
            for year in range(start_year, end_year):
                try:
                    parse(file_descriptor, team, verbose_name, year)
                except Exception as err:
                    restore_backup(filename)
                    raise err
            # do last year weekby week if end_week provided
            if not end_week:
                continue
            for week in range(1, end_week+1):
                try:
                    parse(file_descriptor, team, verbose_name, end_year, week)
                except Exception as err:
                    restore_backup(filename)
                    raise err




# appends new week to master spreadsheet
def add_new_week(week, startyear=2012, endyear=datetime.now().year):
    filename = get_filename(startyear, endyear)

    backup_existing_master(filename)

    with open(filename, 'a') as file_descriptor:
        for team, verbose_name in FRANCHISE_DICT.items():
            if (team == "buf" or team == "cin"): print("skipping because bad data")
            else:
                try:
                    parse(file_descriptor, team, verbose_name, endyear, week)
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
            build_master(START_YEAR, END_YEAR, WEEK)
        except KeyboardInterrupt:
            print('Interrupted')
            restore_backup(get_filename(START_YEAR, END_YEAR))
    else:
        if WEEK is None:
            print('Must specify an end week if not rebuilding/restoring master')
            sys.exit(errno.EINVAL)
        try:
            add_new_week(WEEK, START_YEAR, END_YEAR)
        except KeyboardInterrupt:
            print('Interrupted')
            restore_backup(get_filename(START_YEAR, END_YEAR))
