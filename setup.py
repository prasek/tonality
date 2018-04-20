"""
Usage: python setup.py py2app
"""

from setuptools import setup

NAME = 'Audio Classifier'
APP = ['main.py']
DATA_FILES = ['features']
OPTIONS = {
    'packages':['sklearn', 'librosa', 'resampy', 'llvmlite'],
    'plist': {
        'CFBundleName': NAME,
        'CFBundleDisplayName': NAME,
        'CFBundleGetInfoString': 'Classifying audio as music or speech.',
        'CFBundleIdentifier': 'org.prasek.osx.audioclassifier',
        'CFBundleVersion': '0.0.1',
        'CFBundleShortVersionString': '0.0.1',
    }
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
