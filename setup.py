from distutils.core import setup

import setuptools

_ = setuptools.__version__

setup(
    name='junkyard',
    version='1.0',
    packages=[
        'junkyard',
        'junkyard/maintained', 'junkyard/unmaintained',
        'junkyard/maintained/distributions',
        'junkyard/maintained/preprocessing',
        'junkyard/maintained/sports',
    ],
    install_requires=['numpy', 'scipy', 'tensorflow>=2.0.0'],
    url='https://turingequations.com',
    license='GNU GENERAL PUBLIC LICENSE 3',
    author='mdalvi',
    author_email='milind.dalvi@turingequations.com',
    description=''
)
