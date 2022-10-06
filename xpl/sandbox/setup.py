import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-model',
    'xpl-data',
    'xpl-infrastructure-storage',
    'matplotlib>=3.3.3',
    'torch>=1.7.1',
    'torchvision',
    'nltk>=3.5',
    'pandas>=1.2.1',
    'numpy>=1.19.5',
    'pytz>=2019.3',
    'visdom>=0.1.8.9',
    'inquirer>=2.7.0',
    'torchvision>=0.8.2',
    'Pillow>=8.1.2',
    'protobuf>=3.15.6',
    'soundfile'
]

LIBRARY_NAME = 'xpl-sandbox'
LIBRARY_PATH = 'xpl.sandbox'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH,
              f'{LIBRARY_PATH}.Detection',
              f'{LIBRARY_PATH}.Flow',
              f'{LIBRARY_PATH}.MNIST'],
    version='0.1.6',
    description='A sandbox to wire training training processes.',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
