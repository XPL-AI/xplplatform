import json
import requests

from xpl.concept import config


BABEL_NET_URI = config['babel_net_uri']
BABEL_NET_KEY = config['babel_net_key']


def search(lemma):
    result = requests.get(f'{BABEL_NET_URI}/getSynsetIds?lemma={lemma}&searchLang=EN&key={BABEL_NET_KEY}')

    if result.status_code == 200:
        return json.loads(result.text)

    raise Exception(result.status_code, result.text)
