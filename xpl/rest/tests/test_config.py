import os

ENV = 'DEV'

if ENV == 'DEV':
    BASE_URL = 'http://127.0.0.1:8000'
    HEADERS = {'ClientKey': '204a24293e37496aa', 'ClientSecret': ''}

if ENV == 'STAGE':
    BASE_URL = ''
    HEADERS = {'ClientKey': '204a24293e37496aa', 'ClientSecret': ''}

ABS_PATH = os.path.abspath(__file__)
BASE_DIR = os.path.dirname(ABS_PATH)
SAMPLES_DIR = 'samples'
