import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'sklearn',
    'inquirer',
    'visdom',
    'pandas',
    'numpy',
    'scipy',
    'google-cloud-firestore',
    'pytest',
    'pydantic'
]

LIBRARY_NAME = 'xpl-user'
LIBRARY_PATH = 'xpl.user'

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
