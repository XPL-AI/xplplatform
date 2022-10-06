import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'torch',
    'torchvision',
    'torchaudio',
    'transformers',
    'datasets',
    'efficientnet_pytorch',
    'matplotlib',
    'soundfile'
]

LIBRARY_NAME = 'xpl-pretrain'
LIBRARY_PATH = 'xpl.pretrain'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH],
    version='0.1.6',
    description='A unit to handle users.',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
