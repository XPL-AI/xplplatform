import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-backend',
    'xpl-infrastructure-config',
    'python-multipart',
    'fastapi>=0.62.0',
    'uvicorn',
    'gunicorn',
    'uvloop'
]

LIBRARY_NAME = 'xpl-rest-backend'
LIBRARY_PATH = 'xpl.rest.backend'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH,
              f'{LIBRARY_PATH}.endpoints',
              f'{LIBRARY_PATH}.routers'],
    version='0.1.7',
    description='API first Python library',
    author='I K',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
