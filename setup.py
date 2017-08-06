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
    'h5py',
    'ipython',
    'keras',
    'netCDF4',
    'numpy',
    'osgeo',
    'owslib',
    'pandas',
    'pyproj',
    'requests',
    'suds-py3',
    'tensorflow',
    ],

setup(name='groundwater timenet',
      version=version,
      description="Dutch groundwater prediction (interpolation) model.",
      long_description=long_description,
      # Get strings from http://www.python.org/pypi?%3Aaction=list_classifiers
      classifiers=['Programming Language :: Python',
                   'Development Status :: 2 - Pre-Alpha',
                   'License :: OSI Approved :: '
                   'GNU General Public License v3 or later (GPLv3+)',
                   'Programming Language :: Python :: 3 :: Only'
                   ],
      keywords=[],
      author='Roel van den Berg',
      author_email='roelvdberg@gmail.com',
      url='',
      license='GPLv3',
      packages=['groundwater_timenet'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      entry_points={
          'console_scripts': [
          ]},
      )
