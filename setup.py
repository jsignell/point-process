from setuptools import setup, find_packages

setup(name='pointprocess',
      version='0.3.1',
      description='Tools for reading ungridded point process data',
      url='http://github.com/jsignell/point-process',
      author='Julia Signell',
      author_email='jsignell@gmail.com',
      license='MIT',
      packages=['pointprocess'],
      zip_safe=False,
      # If any package contains *.r files, include them:
      package_data={'': ['*.r', '*.R']},
      include_package_data=True)
