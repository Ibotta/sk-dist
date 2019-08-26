"""
Run setup
"""

from setuptools import setup, find_packages

setup(name='sk-dist',
      version='0.0.1',
      description='Distributed scikit-learn meta-estimators with PySpark',
      classifiers=[
        'Development Status :: 4 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Machine Learning'
      ],
      author='Ibotta Inc.',
      author_email='machine_learning@ibotta.com',
      license='Apache 2.0',
      packages=find_packages(),
      install_requires=[
          'pandas>=0.19.0',
          'numpy>=1.17.0',
          'scipy>=1.3.1',
          'scikit-learn>=0.21.3',
          'six>=1.5',
          'joblib>=0.11'         
      ])
