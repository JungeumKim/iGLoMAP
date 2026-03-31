from setuptools import setup
from setuptools import find_packages

setup(
    name='iglo',
    version='0.1.0',
    author='Jungeum Kim, Xiao Wang',
    description='iGLoMAP: Inductive Global and Local Manifold Approximation and Projection',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.9.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'numba>=0.54.0',
        'PyYAML>=5.3.0',
    ],
)


