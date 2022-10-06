import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

#TODO: update requirements and readme files

install_requires = [
    'xpl-infrastructure-config',
    'xpl-dataset',
    'torch',
    'pydantic'
]

LIBRARY_NAME = 'xpl-train'
LIBRARY_PATH = 'xpl.train'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: working_dir},
    packages=[LIBRARY_PATH],
    version='0.1.6',
    description='Training unit is responsible for hosting all the training related source codes',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(working_dir, library_name=LIBRARY_NAME)
