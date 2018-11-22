# -*- coding: utf-8 -*-
import codecs

from setuptools import setup, find_packages

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name='ocrd_typegroups_classifier',
    version='0.0.1',
    description='Typegroups classifier for OCR',
    long_description=README,
    author='Matthias Seuret',
    author_email='seuretm@users.noreply.github.com',
    url='https://github.com/seuretm/ocrd_typegroups_classifier',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    install_requires=[
        'ocrd >= 0.11.0',
        'torch',
        'pandas',
        'scikit-image',
        'torchvision',
    ],
    package_data={
        '': ['*.pth'],
    },
    entry_points={
        'console_scripts': [
            'typegroups-identifier=ocrd_typegroups_classifier.cli.simple:cli',
        ]
    },
)
