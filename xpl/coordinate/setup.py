import os
import subprocess
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-model',
    'xpl-dataset',
    'xpl-train',
    'xpl-measurer',
    'xpl-user',
    'xpl-task',
    'certifi',
    'matplotlib',
    'torch',
    'torchvision',
    'nltk',
    'pandas',
    'numpy',
    'pytz',
    'visdom',
    'inquirer',
    'torchvision',
    'Pillow',
    'protobuf',
    'google-api-core'
]

LIBRARY_NAME = 'xpl-coordinate'
LIBRARY_PATH = 'xpl.coordinate'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH],
    version='0.1.6',
    description='This is the executes the training procedure which depends on data and train among other things.',
    author='AR, IK',
    license='XPL',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
