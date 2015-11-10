try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'resonator_tools',
    'author': 'Sebastian Probst',
    'url': 'https://github.com/sebastianprobst/resonator_tools',
    'download_url': 'https://github.com/sebastianprobst/resonator_tools',
    'author_email': 'seb.probst@gmail.com',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['resonator_tools'],
    'scripts': [],
    'name': 'resonator_tools'
}

setup(**config)