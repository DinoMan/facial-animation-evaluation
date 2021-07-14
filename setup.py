from setuptools import setup

setup(name='fae',
      version='0.1',
      description='Facial animation evaluation suite',
      packages=['fae'],
      package_dir={'fae': 'fae'},
      package_data={'fae': ['resources/*']},
      install_requires=[
          'torch',
          'numpy',
          'opencv-python',
          'jiwer',
          'gdown',
      ],
      zip_safe=False)
