import json
import config
import requests


cookies_file = '../.cookies'
file_name = '../.creds'
"""
should exist in project root and contain  the following json format 
{
    "username": "wendrickje",
    "password": "wouldntyouliketoknow"
}
"""

def get_creds():
    """
    read hidden .creds file
    """
    creds_file = None
    try:
        creds_file = open(file_name)
    except:
        raise Exception('.creds file is missing. It should exist in project root and contain json object with username and password')
    creds = json.loads(creds_file.read())
    username = creds['username']
    password = creds['password']
    return username, password

def login():
    username, password = get_creds()
    print(username+password)
    payload = {
        'login': 1,
        'username': username,
        # yes, officefootballpool.com uses plain text passwords
        'password': password,
        'gotopage': f'{config.PICKS_PAGE}?thispoolid={config.FOOTBALL_POOL_ID}',
        'poolid': config.FOOTBALL_POOL_ID,
        'supressAlerts': 1
    }
    headers = {
        **config.USER_AGENT
    }

    url = f'{config.OFFICE_FOOTBALL_POOL_URL}/{config.PICKS_PAGE}'
    session = requests.Session()
    response = session.post(url, data=payload, headers=headers)
    response.raise_for_status()
    response_headers = response.headers
    cookies = response_headers['Set-Cookie']
    with open(cookies_file, 'w') as c:
        c.write(cookies)
    
    return session



if __name__ == '__main__':
    session = login()
    





