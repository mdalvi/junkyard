"""
https://github.com/cloudtools/troposphere/blob/master/setup.py
https://dzone.com/articles/executable-package-pip-install
"""
import os

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def file_contents(file_name):
    """Given a file name to a valid file returns the file object."""
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(curr_dir, file_name)) as the_file:
        contents = the_file.read()
    return contents


def get_version():
    curr_dir = os.path.abspath(os.path.dirname(__file__))
    with open(curr_dir + "/junkyard/__init__.py", "r") as init_version:
        for line in init_version:
            if "__version__" in line:
                return str(line.split("=")[-1].strip(" ")[1:-2])


setup(
    name="junkyard",
    version="0.0.1",
    author="mdalvi",
    author_email="milind.dalvi14@gmail.com",
    description="A bunch of utilities that I use and don't use during development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mdalvi/junkyard",
    license="New BSD license",
    packages=[
        'junkyard',
        'junkyard.maintained',
        'junkyard.maintained.common',
        'junkyard.maintained.distributions',
        'junkyard.maintained.loss_functions',
        'junkyard.maintained.preprocessing',
        'junkyard.maintained.sports',
        'junkyard.unmaintained',
        'junkyard.unmaintained.visualization',
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=file_contents("requirements.txt"),
    test_suite="tests",
    use_2to3=True,
)
