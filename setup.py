import os
from setuptools import setup

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md')) as f:
    long_description = f.read()

setup(
    name="rlshield",
    version="0.1",
    description="Shielding on POMDPs with Storm",
    long_description=long_description,
    long_description_content_type='text/markdown',

    packages=["rlshield"],
    install_requires=[
        "stormpy>=1.4.0"
    ],
    python_requires='>=3',
)
