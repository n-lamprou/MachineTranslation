from setuptools import setup
from setuptools import find_packages

setup(name='ComputerViz-DL',
      version='0.1',
      description='Using Deep Learning for computer vision',
      url='https://github.com/n-lamprou',
      author='Nik Lamprou',
      author_email='nikolaos.lamprou1@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'matplotlib == 2.2.2',
            'numpy == 1.14.2',
            'pandas == 0.22.0',
            'python-utils == 2.3.0',
            'scikit-learn == 0.19.1',
            'scipy == 1.0.0',
            'seaborn == 0.8.1',
            'sklearn == 0.0',
            'joblib == 0.11',
            'jupyter == 1.0.0',
	    'six>=1.9.0',
            'pyyaml',
            'h5py',
	    'Keras == 2.0.0',
	    'tensorflow == 1.12.0',
	    'tensorboard == 1.12.0',
	    'scikit-image == 0.13.1',
	    'notebook == 5.7.2',
            ]
      )

