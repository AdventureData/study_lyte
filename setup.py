#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = ["pandas > 1.2.0", "pandas< 2.0.0", 
        "scipy>=1.8.0", "scipy<2.0.0"]
test_requirements = ['pytest>=3', ]

setup(
    author="Micah Johnson ",
    author_email='info@adventuredata.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',

    ],
    description="Package for doing analysis with Lyte Probe data",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='study_lyte',
    name='study_lyte',
    packages=find_packages(include=['study_lyte', 'study_lyte.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AdventureData/study_lyte',
    version='0.3.0',
    zip_safe=False,
)
