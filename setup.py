from setuptools import setup, find_packages

# pip install -e .
setup(
    name='prefect_tutorial',
    version='1.0',
    packages=find_packages(),
    install_requires=[line.strip() for line in open('requirements.txt')],
)
