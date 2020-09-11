#!/usr/bin/env python
from os import path
from setuptools import setup, find_packages
from pathlib import Path
from setuptools import setup
import re

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

basepath = Path(__file__).parent
install_requires = (basepath / "requirements.txt").read_text()
install_requires = re.sub(r"^(git\+https.*)egg=([_\w-]+)$", r"\2@\1", install_requires, 0, re.MULTILINE)

setup(
    name='OpenNIR-XPM',
    description='OpenNIR: A Complete Neural Ad-Hoc Ranking Pipeline (Experimaestro version)',
    author="Sean MacAvaney (adapted by Benjamin Piwowarski for experimaestro)",
    author_email="b@piwowarski.fr",
    long_description=long_description,
    long_description_content_type='text/markdown',
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    url="https://github.com/bpiwowar/OpenNIR-xpm",
    project_urls={
        "Documentation": "https://github.com/bpiwowar/OpenNIR-xpm",
        "Source": "https://github.com/bpiwowar/OpenNIR-xpm"
    },
    install_requires=install_requires,
    entry_points={
    },
    python_requires='>=3.7',
)
