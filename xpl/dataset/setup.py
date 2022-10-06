import os
import subprocess
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-data',
    'xpl-task',
    'torch',
    'torchvision',
    'torchaudio',
    'torchtext',
    'Pillow',
    'imgaug'
]

LIBRARY_NAME = 'xpl-dataset'
LIBRARY_PATH = 'xpl.dataset'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH],
    version='0.1.6',
    description='This is the repository that deals with datasets from Software 2.0 prespective.',
    author='AR, IK',
    license='XPL',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
