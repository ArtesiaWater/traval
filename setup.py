from setuptools import setup, find_packages

try:
    import pypandoc
    l_d = pypandoc.convert('README.md')
except:
    l_d = ''

# Get the version.
version = {}
with open("version.py") as fp:
    exec(fp.read(), version)

setup(
    name='traval',
    version=version['__version__'],
    description='traval module by Artesia',
    long_description=l_d,
    url='https://github.com/ArtesiaWater/traval',
    author='Artesia',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7'
    ],
    platforms='Windows, Mac OS-X, *nix',
    install_requires=['numpy>=1.18', 'matplotlib>=3.0', 'pandas>=0.25', ],
    packages=find_packages(exclude=[])
)
