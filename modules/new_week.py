import sys
from login import login
import numpy as np
from lxml import html
import re
from datetime import datetime, timedelta
import pandas as pd
import string
import config
import os
import argparse
from build_dataset import COL_NAMES as MASTER_COLUMNS

def download_picks_page(year, week, override=False):
    game_doc = f'../game_docs/games{year}-week{week}.html'
    already_downloaded = override
    if not override:
        already_downloaded = os.path.exists(game_doc)
    h = ''
    if not already_downloaded:
        session = login()

        headers = {
            **config.USER_AGENT,
        }

        url = f'{config.OFFICE_FOOTBALL_POOL_URL}/{config.PICKS_PAGE}'
        response = session.get(url, headers=headers)
        response.raise_for_status()
        h = html.fromstring(response.text)
        printer_friendly = h.xpath('//a[@title="Blank Printable Version"]')
        printer_friendly = printer_friendly[0]
        printer_friendly_page = printer_friendly.attrib['href']

        url = f'{config.OFFICE_FOOTBALL_POOL_URL}/{printer_friendly_page}'
        response = session.get(url, headers=headers)

        with open(game_doc, 'w') as fi:
            fi.write(response.text)
        h = response.text
    else:
        with open(game_doc) as fi:
            h = fi.read()
    return html.fromstring(h)


def parse_to_dateframe(h, year, week):
    table = h.xpath('//*[@id="pickem"]')
    table = h.xpath('//*[@id="pickem"]')
    table = table[0]
    tds = table.xpath('//td')
    text = [re.sub(r'\s{2}', '', td.text_content().strip()) for td in tds][:-2]
    print(text)
    step = 4
    df = pd.DataFrame()
    for i, start in enumerate(range(0, len(text), step)):
        end = start + step
        away_team, game_time, home_team = text[start + 1 : end]
        game_day = game_time[:3]
        game_time = datetime.strptime(
            game_time + f' {year} week{week}', '%a %H:%M %p %G week%V'
        )
        # need to move monday ahead for sorting
        if game_time.weekday() == 0:
            game_time += timedelta(days=7)
        [away_team, away_spread] = away_team.split(')')
        away_spread = float(away_spread)
        away_team = away_team.split('(')[0].strip().lower()
        [home_team, home_spread] = home_team.split(')')
        home_spread = float(home_spread)
        home_team = home_team.split('(')[0].strip().lower()

        row = [
            {
                'home_team': home_team,
                'home_spread': home_spread,
                'game_day': game_day,
                'game_time': game_time.time(),
                'sort_key': game_time,
                'away_team': away_team,
                'away_spread': away_spread,
                'boxscore_url': string.ascii_uppercase[i],
            }
        ]
        df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
    df = df.sort_values('sort_key')
    df = df.drop(columns=['sort_key'])

    return df


def rename_columns(df: pd.DataFrame):
    """
    rename to expected columns
    """
    df = df.rename(
        columns={
            'home_team': 'verbose_name',
            'game_day': 'day',
            'game_time': 'time',
            'away_team': 'opponent',
        }
    )
    return df


def expand_teams(df: pd.DataFrame):
    """
    every team needs to be represented.
    this will add all opponents as teams
    """
    df2 = df[df.columns]
    df2['temp'] = df2['opponent']
    df2['opponent'] = df2['verbose_name']
    df2['verbose_name'] = df2['temp']

    df2['temp_spread'] = df2['away_spread']
    df2['away_spread'] = df2['home_spread']
    df2['home_spread'] = df2['temp_spread']
    df2['at'] = '@'
    df2 = df2.drop(columns=['temp', 'temp_spread'])
    return pd.concat([df, df2], ignore_index=True)


def map_team_names(df):
    def team_mapper(team):
        return config.TEAM_MAPPER.get(team, [None, None])

    df['team'] = df['verbose_name'].apply(lambda team: team_mapper(team)[1])
    df['verbose_name'] = df['verbose_name'].apply(lambda team: team_mapper(team)[0])
    df['opponent'] = df['opponent'].apply(lambda team: team_mapper(team)[0])
    return df


def add_placeholders(df):
    df = pd.merge(
        df,
        pd.DataFrame([config.PLACE_HOLDERS] * len(df)),
        left_index=True,
        right_index=True,
    )
    return df


def get_vegas_close_line(df):
    def concat_spread_name(row):
        name, spread = row
        return f'{name} {spread}'

    df['Vegas_Line_Close'] = np.where(
        df['home_spread'] < df['away_spread'],
        df[['verbose_name', 'home_spread']].apply(concat_spread_name, axis='columns'),
        df[['opponent', 'away_spread']].apply(concat_spread_name, axis='columns'),
    )
    return df


def drop_and_reorder_columns(df):
    """
    does order matter? YES
    """
    df = df.drop(columns=['home_spread', 'away_spread'])
    df = df[MASTER_COLUMNS]
    return df


def new_week(year, week):
    h = download_picks_page(year, week)
    df = parse_to_dateframe(h, year, week)
    df['year'] = year
    df['week'] = week
    df['at'] = ''
    df = rename_columns(df)
    df = expand_teams(df)
    df = map_team_names(df)
    df = add_placeholders(df)
    df = get_vegas_close_line(df)
    df = drop_and_reorder_columns(df)
    output = f'../game_docs/games{year}-week{week}.csv'
    print(f'Success! Wrote to {output}')
    df.to_csv(output, index=False)
    return df


if __name__ == '__main__':
    parsed = argparse.ArgumentParser()
    parsed.add_argument('-y', '--year', type=int, default=datetime.now().year)
    parsed.add_argument('-w', '--week', type=int, default=1)
    args = parsed.parse_args()
    week, year = args.week, args.year

    new_week(year, week)
