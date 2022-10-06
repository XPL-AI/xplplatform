import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-infrastructure-config',
    'xpl-concept',
    'xpl-data',
    'pytest'
]

LIBRARY_NAME = 'xpl-annotation'
LIBRARY_PATH = 'xpl.annotation'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: working_dir},
    packages=[LIBRARY_PATH],
    version='0.1.7',
    description='An annotation unit main library.',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(working_dir, library_name=LIBRARY_NAME)