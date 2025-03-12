__author__ = "Maxim Ziatdinov"
__maintainer__ = "Maxim Ziatdinov"
__email__ = "maxim.ziatdinov@gmail.com"

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(module_dir, 'neurobayes/__version__.py')) as f:
    __version__ = f.read().split("'")[1]

if __name__ == "__main__":
    setup(
        name='NeuroBayes',
        python_requires='>=3.9',
        version=__version__,
        description='Fully and Partially Bayesian Neural Networks',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ziatdinovmax/NeuroBayes/',
        author='Maxim Ziatdinov',
        author_email='maxim.ziatdinov@gmail.com',
        license='MIT license',
        packages=find_packages(),
        zip_safe=False,
        install_requires=[
            'jax>=0.4.0,<=0.4.31',
            'jaxlib>=0.4.0,<=0.4.31',
            'numpyro>=0.13.0',
            'flax>=0.8.4'
        ],
        classifiers=['Programming Language :: Python',
                     'Development Status :: 3 - Alpha',
                     'Intended Audience :: Science/Research',
                     'Operating System :: POSIX :: Linux',
                     'Operating System :: MacOS :: MacOS X',
                     'Topic :: Scientific/Engineering']
    )
