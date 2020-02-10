"""
Run setup
"""

from setuptools import setup, find_packages
from skdist import __version__

DISTNAME = "sk-dist"
VERSION = __version__
DESCRIPTION = "Distributed scikit-learn meta-estimators with PySpark"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering"
    ]
AUTHOR = "Ibotta Inc."
AUTHOR_EMAIL = "machine_learning@ibotta.com"
LICENSE = "Apache 2.0"
DOWNLOAD_URL = "https://pypi.org/project/sk-dist/#files"
PROJECT_URLS = {
    "Source Code": "https://github.com/Ibotta/sk-dist"
    }
MIN_PYTHON_VERSION = "3.5"
MIN_PANDAS_VERSION = "0.17.0"
MIN_SKLEARN_VERSION = "0.20.0"
MIN_XGBOOST_VERSION = "0.4"
PYARROW_VERSION = "0.15.0"
MIN_PYSPARK_VERSION = "2.4.4"
MIN_PYTESTSPARK_VERSION = "0.4.5"

install_requires = [
    "scikit-learn>={0}".format(MIN_SKLEARN_VERSION),
    "pandas>={0}".format(MIN_PANDAS_VERSION),
    "numpy",
    "scipy",
    "joblib"
]

tests_require = [
    "xgboost>={0}".format(MIN_XGBOOST_VERSION),
    "pyarrow=={0}".format(PYARROW_VERSION),
    "pyspark>={0}".format(MIN_PYSPARK_VERSION),
    "pytest-spark>={0}".format(MIN_PYTESTSPARK_VERSION)
]

def parse_description(description):
    """
    Strip figures and alt text from description
    """
    return "\n".join(
        [
        a for a in description.split("\n")
        if ("figure::" not in a) and (":alt:" not in a)
        ])

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=parse_description(LONG_DESCRIPTION),
      classifiers=CLASSIFIERS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      packages=find_packages(),
      python_requires=">={0}".format(MIN_PYTHON_VERSION),
      install_requires=install_requires,
      tests_require=tests_require,
      extras_require=dict(tests=tests_require)
      )
