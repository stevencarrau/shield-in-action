import os
from setuptools import setup

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name="storm-project-starter-python",
    version="0.1",
    author="Sebastian Junges",
    author_email="sebastian.junges@cs.rwth-aachen.de",
    maintainer="S. Junges",
    maintainer_email="sebastian.junges@cs.rwth-aachen.de",
    description="Shielding on POMDPs with Storm",
    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=["rlshield"],
    install_requires=[
        "stormpy>=1.3.0"
    ],
    python_requires='>=3',
)
