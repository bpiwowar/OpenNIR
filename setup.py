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
    name='OpenNIR_XPM',
    description='OpenNIR: A Complete Neural Ad-Hoc Ranking Pipeline (Experimaestro version)',
    author="Sean MacAvaney (modified by Benjamin Piwowarski)",
    author_email="b@piwowarski.fr",
    long_description=long_description,
    long_description_content_type='text/markdown',
    version='0.1.0',
    packages=find_packages(),
    url="http://opennir.net/",
    project_urls={
        "Documentation": "http://opennir.net/",
        "Source": "https://github.com/bpiwowar/OpenNIR"
    },
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "onir_init_dataset=onir.bin.init_dataset:main",
            "onir_pipeline=onir.bin.pipeline:main",
            "onir_eval=onir.bin.eval:main",
        ],
    },
    python_requires='>=3.6',
)
