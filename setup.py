"""
Setup скрипт для установки пакета
"""

from setuptools import setup, find_packages

setup(
    name="MA-VCG-QMIX",
    version="1.0.0",
    description="MA-VCG + QMIX для справедливого и адаптивного управления ресурсами",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/Dima12101/MA-VCG-QMIX",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.10.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
