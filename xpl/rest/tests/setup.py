import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-infrastructure-config',
    'xpl-infrastructure-storage',
    'requests',
    'pytest',
    'aiohttp',
    'pytest-asyncio',
    'pydantic',
    'tqdm'
]

LIBRARY_NAME = 'xpl-rest-tests'
LIBRARY_PATH = 'xpl.rest.tests'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH],
    version='0.1.7',
    description='Http tests for all http endpoints. High level tests.',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
