import os
from setuptools import setup

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name="storm-project-starter-python",
    version="0.1",
    author="M. Volk",
    author_email="matthias.volk@cs.rwth-aachen.de",
    maintainer="S. Junges",
    maintainer_email="sebastian.junges@cs.rwth-aachen.de",
    url="https://github.com/moves-rwth/storm-project-starter-python",
    description="Starter project for the Python API of Storm via Stormpy",
    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=["stormpy_starter"],
    install_requires=[
        "stormpy>=1.3.0"
    ],
    python_requires='>=3',
)
