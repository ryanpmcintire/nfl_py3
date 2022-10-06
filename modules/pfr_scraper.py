import os
from turtle import home
import requests
from bs4 import BeautifulSoup, Comment
import numpy as np
import pandas as pd

# from user_agent import generate_user_agent
import re
import cchardet  # speeds up encoding just by import?

import config

# todo: finish user agent
headers = {}

base_url = 'http://www.pro-football-reference.com'
COL_NAMES = [
    'week',
    'day',
    'date',
    'time',
    'boxscore_url',
    *config.SHARED_COLS,
]

relevant_headers = ['Won Toss', 'Roof', 'Surface', 'Vegas Line', 'Over/Under']

parser = 'lxml'
terminal_wdith = os.get_terminal_size().columns
terminal_spacer = " "*(terminal_wdith // 10)


def _strip_html(text):
    """
    Strips tags from HTML
    :param text: The text to strip from
    :return: Cleaned text
    """
    tag_re = re.compile(r'<[^>]+>')
    return tag_re.sub('', str(text))


def _comma_replace(text):
    """
    Replace comma with underscore but doesn't disrupt comma separation between columns
    :param text: The text to strip comma from
    :return: Cleaned text
    """
    return text.replace(',', '_')


def parse_boxscore_url(url_tag):
    """
    Parses the URL from the boxscore tag
    :param url_tag: The url tag
    :return: The boxscore url
    """
    soup = BeautifulSoup(url_tag, features=parser)
    return soup.find_all('a', href=True)[0]['href']


def get_page_html(uri, session_object):
    """
    Visits the page url and returns html as bs4 object
    :param uri:
    :session_object:
    """
    page_url = get_full_url(uri)
    res = session_object.get(page_url)
    if '404' in res.url:
        raise Exception("Could not get: " + page_url)
    return BeautifulSoup(res.text, features=parser)

def get_full_url(uri):
    if uri[0] != '/':
        uri = f'/{uri}'
    page_url = f'{base_url}{uri}'
    return page_url


def parse_season(team, verbose_name, year, end_week):
    print('=' * terminal_wdith)
    print(f'{terminal_spacer}{verbose_name} - {year}')
    session_object = requests.Session() # for reusing session
    season_uri = f'teams/{team}/{year}.htm'
    soup = get_page_html(season_uri, session_object)
    parsed = soup.find('table', {'class': 'sortable stats_table', 'id': 'games'})
    tbody = parsed.find('tbody')
    # the left most column is a table header for some stupid reason
    rows = tbody.find_all(['th', 'td'])
    # need to group the content of rows together
    # there are less columns before 1994
    column_len = 25 if int(year) >= 1994 else 22
    grouped_rows = [rows[i : i + column_len] for i in range(0, len(rows), column_len)]
    data = []

    start_week = end_week - 1 if end_week else 0
    if not end_week:
        end_week = 17 if year < 2021 else 18


    for (week, row) in enumerate(grouped_rows[start_week:end_week]):
        print(f'{"- - "*(terminal_wdith // 4)}')
        print(f'{terminal_spacer}week: {week + start_week + 1}')
        # get boxscore url
        if _strip_html(row[1]) == '':
            continue
        boxscore_uri = parse_boxscore_url(str(row[COL_NAMES.index('boxscore_url')]))
        # use boxscore url to get the game stats
        box_score_rows, away_team = parse_boxscore(boxscore_uri, session_object)
        # strip the html, return a list because map is stupid in py 3
        row = list(map(lambda x: _strip_html(x), row))
        row.insert(0, str(year))
        row.insert(1, str(team))
        row.insert(2, str(verbose_name))
        # add the box score rows
        row.extend(box_score_rows)
        # replace the boxscore text with the boxscore url
        row[COL_NAMES.index('boxscore_url') + 1] = boxscore_uri

        snap_count = parse_snap_count(boxscore_uri, session_object)
        row.extend(snap_count['home'])
        row.extend(snap_count['away'])

        defense = parse_defense(boxscore_uri, session_object)
        row.extend(defense)

        special_teams = parse_special_teams(boxscore_uri, session_object, away_team)
        row.extend(special_teams['returns'])
        row.extend(special_teams['kicks'])

        # attach everything to data
        data.append(','.join(row))
    return data


def parse_boxscore(boxscore_uri, session_object):
    # retrieves game stats, referee info, weather, vegas odds
    soup = get_page_html(boxscore_uri, session_object)

    # game info is the weather, score, vegas odds etc
    all_game_info = soup.find('div', {'id': 'all_game_info'})
    # what we need is hidden behind a comment so must do this nonsense
    # this gets the comment and then converts it back into a bs4 object that we can search
    game_comments = BeautifulSoup(
        all_game_info.find(text=lambda text: isinstance(text, Comment)), "html.parser"  # type: ignore
    )
    # Find only relevant fields, strip html, convert to list. Exclude the column header
    th_list = game_comments.find_all('th')
    game_data = []
    i = 1
    for elem in th_list:
        if elem.text in relevant_headers:
            game_data.extend(
                list(map(lambda x: _strip_html(x), game_comments.find_all('td')[i]))
            )
        i = i + 1
    # stats that are meaningful for predicting team performance
    all_team_stats = soup.find('div', {'id': 'all_team_stats'})
    # what we need is hidden behind a comment so must do this nonsense
    # this gets the comment and then converts it back into a bs4 object that we can search
    team_stats_comments = BeautifulSoup(
        all_team_stats.find(text=lambda text: isinstance(text, Comment)), "html.parser"  # type: ignore
    )
    [away], [home] = [[_strip_html(x) for x in team_stats_comments.find_all(name=['th'], attrs={'data-stat': y})] for y in ['vis_stat', 'home_stat']]
    print(f'{terminal_spacer}url: {get_full_url(boxscore_uri)}')
    print(f'{terminal_spacer}away: {away}')
    print(f'{terminal_spacer}home: {home}')
    # strip html, convert to list. Exclude the column header
    team_stats_data = list(
        map(lambda x: _strip_html(x), team_stats_comments.find_all('td'))
    )
    # append everything to game_data
    game_data.extend(team_stats_data)
    return game_data, away

def parse_snap_count(boxscore_uri, session_object):
    """
    Parses the snap counts for off/def/special-teams for both home and away
    :param boxscore_uri:
    :param session_object:
    """
    soup = get_page_html(boxscore_uri, session_object)
    h_offense, h_off_pct, h_defense, h_def_pct, h_spec_tm, h_spec_tm_pct = get_snap_counts(soup, 'all_home_snap_counts')
    a_offense, a_off_pct, a_defense, a_def_pct, a_spec_tm, a_spec_tm_pct = get_snap_counts(soup, 'all_vis_snap_counts')

    h_off_snap_count = calculate_snap_count(h_offense, h_off_pct)
    h_def_snap_count = calculate_snap_count(h_defense, h_def_pct)
    h_st_snap_count = calculate_snap_count(h_spec_tm, h_spec_tm_pct)
    a_off_snap_count = calculate_snap_count(a_offense, a_off_pct)
    a_def_snap_count = calculate_snap_count(a_defense, a_def_pct)
    a_st_snap_count = calculate_snap_count(a_spec_tm, a_spec_tm_pct)

    return {'home': [h_off_snap_count, h_def_snap_count, h_st_snap_count], 'away': [a_off_snap_count, a_def_snap_count, a_st_snap_count]}

def get_snap_counts(soup, id):
    """
    Get the snap counts for a team
    :param soup: a soup object
    :param id: html id for the team
    """
    snap_table = soup.find('div', {'id': id})
    # because what we want is behind a comment
    new_soup = parse_comment(snap_table)
    snap_columns = ['offense', 'off_pct', 'defense', 'def_pct', 'special_teams', 'st_pct']
    return np.asarray([[_strip_html(x).strip('%') for x in new_soup.find_all('td', {'data-stat': y})] for y in snap_columns], dtype=np.float64)

def calculate_snap_count(snaps, pct):
    """
    Calculate the snap count based on the max val found in snap count col
    If max val is not 100., then take the next highest number and divide it by corresponding percentage
    :param snaps: snap column
    :param pct: pct columnn
    """
    if 100. not in snaps:
        max_snap_i = np.argmax(snaps)
        return str(int(round(snaps[max_snap_i] / (pct[max_snap_i] / 100))))
    return str(int(np.max(snaps)))

def parse_defense(boxscore_uri, session_object):
    """
    Sum the defensive totals for both teams.
    Returns an array with the home team stats first, followed by away team stats.
    :param boxscoure_uri:
    :param session_object:
    """
    soup = get_page_html(boxscore_uri, session_object)
    defense = get_defense(soup)
    return group_and_sum(defense)


def get_defense(soup):
    """
    Get defense stats table (this gets data for both the teams)
    :param soup:
    """
    defense_table = soup.find('div', {'id': 'all_player_defense'})
    new_soup = parse_comment(defense_table)
    defense_columns = ['player', 'team', 'def_int', 'def_int_yds', 'def_int_td', 'def_int_long', 'pass_defended', 'sacks', 'tackles_combined', 'tackles_solo', 'tackles_assists', 'tackles_loss', 'qb_hits', 'fumbles_rec', 'fumbles_rec_yds', 'fumbles_rec_td', 'fumbles_forced']
    return parse_stacked_table(new_soup, defense_columns)

def parse_special_teams(boxscore_uri, session_object, away_team):
    """
    Sum the special team kicking and return stats for both teams
    :param boxsocre_uri:
    :param session_object:
    :param away_team: we need to know the away team because these tables are sometimes missing data for one of the teams
    """
    soup = get_page_html(boxscore_uri, session_object)
    returns, kicks = get_special_teams(soup)
    return {'returns': group_and_sum(returns, away_team), 'kicks': group_and_sum(kicks, away_team) }

def get_special_teams(soup):
    """
    Get the special teams kicking and returns tables
    :param soup:
    """
    returns_table = soup.find('div', {'id': 'all_returns'})
    returns_soup = parse_comment(returns_table)
    returns_columns = ['player', 'team', 'kick_ret', 'kick_ret_yds', 'kick_ret_yds_per_ret', 'kick_ret_td', 'kick_ret_long', 'punt_ret', 'punt_ret_yds', 'punt_ret_yds_per_ret', 'punt_ret_td', 'punt_ret_long']
    returns_df = parse_stacked_table(returns_soup, returns_columns)

    kicking_table = soup.find('div', {'id': 'all_kicking'})
    kicking_soup = parse_comment(kicking_table)
    kicking_columns = ['player', 'team', 'xpm', 'xpa', 'fgm', 'fga', 'punt', 'punt_yds', 'punt_yds_per_punt', 'punt_long']
    kicking_df = parse_stacked_table(kicking_soup, kicking_columns)
    
    return returns_df, kicking_df

def parse_stacked_table(soup, columns):
    """
    Some of the data is structured in a stacked table with away team on top and home team on bottom
    :param soup:
    :param columns: list of columns to pull data for - these correspond to a data-stat attribute in the DOM
    """
    # Pandas here because it organizes this table better than numpy with less work
    df = pd.DataFrame([[_strip_html(x) for x in soup.find_all(name=['th', 'td'], attrs={'data-stat': y, 'aria-label': False})] for y in columns]).T.fillna(0).replace('', '0')
    df.rename({v: k for v, k in enumerate(columns)}, axis=1, inplace=True)
    return df

def parse_comment(soup):
    """
    pro-football-reference does weird thing with commented out html
    :param soup:
    """
    return BeautifulSoup(soup.find(text=lambda x: isinstance(x, Comment)), features=parser)

def group_and_sum(dataframe: pd.DataFrame, away_team = None):
    """
    Group by team and sum row data
    Return is ordered as: [home, away]
    :param dataframe:
    :param away_team: only used by parse_special_teams()
    """
    if away_team and dataframe.groupby('team').ngroups < 2:
        print("Missing data for at least one team in special teams table")
        groups = dataframe.groupby('team').groups
        __abbrv, t1= groupby_team(dataframe).pop()

        t1.drop(['team'], axis=1, inplace=True)
        t1 = t1.astype('float').sum().astype('str').to_numpy()

        t2 = np.zeros_like(t1).astype('str')

        return [*t2, *t1] if away_team in groups else [*t1, *t2]
    else:
        (__t1_abbrv, t1_data), (__t2_abbrv, t2_data) = groupby_team(dataframe)
        t1_min = t1_data.index.min()
        t2_min = t2_data.index.min()

        t1_data.drop(['team'], axis=1, inplace=True)
        t1 = t1_data.astype('float').sum().astype('str').to_numpy()

        t2_data.drop(['team'], axis=1, inplace=True)
        t2 = t2_data.astype('float').sum().astype('str').to_numpy()

        return [*t2, *t1] if t1_min < t2_min else [*t1, *t2]

def groupby_team(dataframe: pd.DataFrame):
    return [x for x in dataframe.drop(['player'], axis=1).groupby('team')]