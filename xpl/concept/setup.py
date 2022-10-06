import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl.infrastructure.config',
    'google-cloud-firestore',
    'pydantic',
    'nltk',
    'pandas',
    'pytest',
    'inquirer'
]

LIBRARY_NAME = 'xpl-concept'
LIBRARY_PATH = 'xpl.concept'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: working_dir},
    packages=[LIBRARY_PATH],
    version='0.1.7',
    description='A concept management unit main library.',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(working_dir, library_name=LIBRARY_NAME)
