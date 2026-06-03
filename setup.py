"""
NeuralForge Setup
"""

from setuptools import setup, find_packages

setup(
    name="neuralforge",
    version="0.1.0",
    description="A from-scratch language model - no external model dependencies",
    author="NeuralForge Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    entry_points={
        'console_scripts': [
            'neuralforge-train=train:main',
            'neuralforge-generate=generate:main',
        ],
    },
)
