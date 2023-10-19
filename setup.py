
from setuptools import setup, find_packages

VERSION = '1.0.0' 
DESCRIPTION = 'Dimension-reduction evaluation toolbox'
LONG_DESCRIPTION = 'Unsupervised benchmarking and supervised explanatory evaluation measures for dimensionality reduction of single-cell data'

setup(
        name='ViScore', 
        version=VERSION,
        author='David Novak',
        author_email="<davidnovakcz@hotmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'pynndescent', 'numba'],
        keywords=['python', 'LCMC', 'RNX', 'xNPE'],
        classifiers= [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Bioinformatics",
            "Programming Language :: Python :: 3"
        ]
)
