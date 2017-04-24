# -*- coding: utf-8 -*-

from setuptools import setup
import monkeypatch_setuptools  # NOQA

version = '0.1.dev0'

long_description = '\n\n'.join([
    open('README.md').read(),
    open('CREDITS.md').read(),
    open('CHANGES.md').read(),
    ])

install_requires = [
    'keras',
    'netcdf4',
    'numpy',
    'osgeo',
    'pandas',
    'requests',
    'suds-py3',
    'tensorflow',
    ],

setup(name='ggmn-site',
      version=version,
      description="TODO",
      long_description=long_description,
      # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=['Programming Language :: Python',
                   'Framework :: Django',
                   ],
      keywords=[],
      author='Roel van den Berg',
      author_email='roelvdberg@gmail.com',
      url='',
      license='MIT',
      packages=['groundwater_timenet'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      entry_points={
          'console_scripts': [
          ]},
      )
