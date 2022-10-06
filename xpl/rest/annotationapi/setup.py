import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-infrastructure-config',
    'xpl-user',
    'xpl-annotation',
    'python-multipart',
    'fastapi>=0.62.0',
    'uvicorn',
    'gunicorn',
    'uvloop',
    'jinja2',
    'aiofiles'
]

LIBRARY_NAME = 'xpl-rest-annotationapi'
LIBRARY_PATH = 'xpl.rest.annotationapi'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH,
              f'{LIBRARY_PATH}.routers'],
    version='0.1.7',
    description='Annotation api.',
    author='I K',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
