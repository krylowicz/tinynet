from pathlib import Path
from setuptools import setup, find_packages

long_description = Path(Path(__file__).parent / 'README.md').read_text()

setup(
  name='tinynet',
  version='0.1.0',
  description='a tiny deep learning framework',
  long_description=long_description,
  long_description_content_type='text/markdown',
  author='@krylowicz',
  packages=find_packages(include=['tinynet', 'tinynet.*']),
  install_requires=[
    'numpy>=1.22.3',
  ],
  extras_require={
    'dev': ['pytest>=7.0.1', 'torch>=1.11.0', 'tqdm>=4.63.0']
  }
)