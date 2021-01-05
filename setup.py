from distutils.core import setup

setup(
    name='junkyard',
    version='1.0',
    packages=[
        'maintained', 'unmaintained',
        'maintained/distributions',
        'maintained/preprocessing',
    ],
    install_requires=['numpy', 'scipy'],
    url='https://turingequations.com',
    license='GNU GENERAL PUBLIC LICENSE 3',
    author='mdalvi',
    author_email='milind.dalvi@turingequations.com',
    description=''
)
