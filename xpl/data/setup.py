import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-infrastructure-config',
    'xpl-infrastructure-storage',
    'nltk',
    'pandas',
    'visdom',
    'inquirer',
    'pytz',
    'google-cloud-bigquery',
    'google-cloud-bigquery-storage',
    'pytest',
    'pydantic',
    'pyarrow',
    'torch',
    'torchvision',
    'torchaudio',
    'certifi',
    'beautifulsoup4',
    'matplotlib',
    'bs4'
]

LIBRARY_NAME = 'xpl-data'
LIBRARY_PATH = 'xpl.data'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: working_dir},
    packages=[LIBRARY_PATH],
    version='0.1.6',
    description='Data management unit',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(working_dir, library_name=LIBRARY_NAME)
