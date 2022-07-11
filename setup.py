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
    'numpy',
    'requests',
  ],
  extras_require={
    'testing': ['pytest', 'torch', 'tqdm'],
    'gpu': ['pyopencl']
  }
)
