import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xplai'
]

LIBRARY_NAME = 'xpl-test'
LIBRARY_PATH = 'xpl.test'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: './'},
    packages=[LIBRARY_PATH],
    version='0.1.6',
    description='A unit to handle test.',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(from_directory=working_dir, library_name=LIBRARY_NAME)
