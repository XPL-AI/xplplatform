import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-infrastructure-config',
    'xpl-infrastructure-storage',
    'xpl-user',
    'xpl-data',
    'xpl-task',
    'python-multipart',
    'fastapi>=0.62.0',
    'uvicorn',
    'gunicorn',
    'uvloop'
]

LIBRARY_NAME = 'xpl-rest-dataapi'
LIBRARY_PATH = 'xpl.rest.dataapi'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH,
              f'{LIBRARY_PATH}.routers'],
    version='0.1.7',
    description='Data api.',
    author='I K',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
