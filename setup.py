from setuptools import setup, find_packages

setup(name='dependency_parsers',
      version='1.0',
      description='Evaluation of a Semi Supervised dependency parser using First and Second Order Expectations',
      author='Antonia Boca',
      url='https://github.com/antoniaboca/semi-supervised-dependency-parser',
      install_requires=[
          pyconll, torch, torchvision, torchaudio
      ],
      packages=find_packages(),
      )