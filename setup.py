from setuptools import setup, find_packages

setup(
    name='mindl',
    version='0.1.0',
    url='https://github.com/jorahn/minDL.git',
    author='Jonathan Rahn',
    author_email='jonathan.rahn@42digital.de',
    description='Minimalist Deep Learning Experiments in PyTorch & TensorFlow',
    packages=find_packages(),
    install_requires=['torch == 1.9.0', 'torchvision == 0.10.0', 'tensorflow == 2.6.0'],
)
