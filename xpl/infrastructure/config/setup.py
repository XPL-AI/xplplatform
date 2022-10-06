import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'google-cloud-firestore',
    'pydantic'
]

LIBRARY_NAME = 'xpl-infrastructure-config'
LIBRARY_PATH = 'xpl.infrastructure.config'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: working_dir},
    packages=[LIBRARY_PATH],
    version='0.1.7',
    description='A library responsible for the configuration management infrastructure-wide.',
    author='IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(working_dir, library_name=LIBRARY_NAME)
