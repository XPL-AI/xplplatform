import os
from setuptools import setup

from xpl import build_tools


working_dir = os.path.dirname(__file__)
build_tools.cleanup(working_dir, include_egg_info=True)

install_requires = [
    'xpl-infrastructure-config',
    'xpl-infrastructure-storage',
    'numpy',
    'torch',
    'pandas',
    'visdom',
    'inquirer',
    'google-cloud-storage',
    'gcloud',
    'soundfile'
]

LIBRARY_NAME = 'xpl-model'
LIBRARY_PATH = 'xpl.model'

setup(
    name=LIBRARY_NAME,
    package_dir={LIBRARY_PATH: working_dir},
    packages=[LIBRARY_PATH,
              f'{LIBRARY_PATH}'],
    version='0.1.6',
    description='Model Management Unit is a stand-alone module that preserves pytorch models, '
                'as well as their histories and their measurements. Ideally, it has an API that stores, '
                'loads and updates models while keeping track of their measurements',
    author='AR, IK',
    license='MIT',
    install_requires=install_requires
)

build_tools.move_distributions(working_dir, library_name=LIBRARY_NAME)
