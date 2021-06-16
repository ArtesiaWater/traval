from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'readme.md'), encoding='utf-8') as f:
    l_d = f.read()

# Get the version.
version = {}
with open("traval/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='traval',
    version=version['__version__'],
    description='Python package for applying automatic error detection '
                'algorithms to timeseries. Create custom error detection '
                'algorithms to support data validation workflows.',
    long_description=l_d,
    long_description_content_type='text/markdown',
    url='https://github.com/ArtesiaWater/traval',
    author='Artesia',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Hydrology',
    ],
    platforms='Windows, Mac OS-X, *nix',
    install_requires=['numpy>=1.18',
                      'scipy>=1.1',
                      'matplotlib>=3.0',
                      'pandas>=0.25'],
    packages=find_packages(exclude=[])
)
