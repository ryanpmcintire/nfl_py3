import requests
from bs4 import BeautifulSoup, Comment

# from user_agent import generate_user_agent
import re
import cchardet  # speeds up encoding just by import?

# todo: finish user agent
headers = {}

base_url = 'http://www.pro-football-reference.com'
COL_NAMES = [
    'week',
    'day',
    'date',
    'time',
    'boxscore_url',
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
    'Weather',
    'Vegas_Line_Close',
    'Over/Under',
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
]

relevant_headers = ['Won Toss', 'Roof', 'Surface', 'Vegas Line', 'Over/Under']

parser = 'lxml'


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


# this is fragile af since it relies on html tags not changing


def parse_season(team, verbonse_name, year, endWeek):
    # for reusing session
    session_object = requests.Session()
    season_url = f'http://www.pro-football-reference.com/teams/{team}/{year}.htm'
    res = session_object.get(season_url)
    if '404' in res.url:
        raise Exception(f'No data found for team {team} in year {year}')
    soup = BeautifulSoup(res.text, features=parser)
    parsed = soup.find('table', {'class': 'sortable stats_table', 'id': 'games'})
    tbody = parsed.find('tbody')
    # the left most column is a table header for some stupid reason
    rows = tbody.find_all(['th', 'td'])
    # need to group the content of rows together
    # there are less columns before 1994
    column_len = 25 if int(year) >= 1994 else 22
    grouped_rows = [rows[i : i + column_len] for i in range(0, len(rows), column_len)]
    data = []

    if not endWeek:
        endWeek = 17 if year < 2021 else 18

    startWeek = endWeek - 1 if endWeek else 0
    for row in grouped_rows[startWeek:endWeek]:
        # get boxscore url
        if _strip_html(row[1]) == '':
            continue
        box_score_uri = parse_boxscore_url(str(row[COL_NAMES.index('boxscore_url')]))
        # use boxscore url to get the game stats
        box_score_rows = parse_boxscore(box_score_uri, session_object)
        # strip the html, return a list because map is stupid in py 3
        row = list(map(lambda x: _strip_html(x), row))
        row.insert(0, str(year))
        row.insert(1, str(team))
        row.insert(2, str(verbonse_name))
        # add the box score rows
        row.extend(box_score_rows)
        # replace the boxscore text with the boxscore url
        row[COL_NAMES.index('boxscore_url') + 1] = box_score_uri
        # attach everything to data

        data.append(','.join(row))
    return data


# retrieves game stats, referee info, weather, vegas odds


def parse_boxscore(box_score_uri, session_object):
    boxscore_url = base_url + box_score_uri
    res2 = session_object.get(boxscore_url)
    if '404' in res2.url:
        raise Exception("Could not get box score at url: " + boxscore_url)
    soup2 = BeautifulSoup(res2.text, features=parser)

    # game info is the weather, score, vegas odds etc
    all_game_info = soup2.find('div', {'id': 'all_game_info'})
    # what we need is hidden behind a comment so must do this nonsense
    # this gets the comment and then converts it back into a bs4 object that we can search
    game_comments = BeautifulSoup(
        all_game_info.find(text=lambda text: isinstance(text, Comment)), "html.parser"
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
    all_team_stats = soup2.find('div', {'id': 'all_team_stats'})
    # what we need is hidden behind a comment so must do this nonsense
    # this gets the comment and then converts it back into a bs4 object that we can search
    team_stats_comments = BeautifulSoup(
        all_team_stats.find(text=lambda text: isinstance(text, Comment)), "html.parser"
    )
    # strip html, convert to list. Exclude the column header
    team_stats_data = list(
        map(lambda x: _strip_html(x), team_stats_comments.find_all('td'))
    )
    # append everything to game_data
    game_data.extend(team_stats_data)
    # leaving this in because its a good way to know which links cause problems
    print(boxscore_url)
    print(game_data)
    return game_data
