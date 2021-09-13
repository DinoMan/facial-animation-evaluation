from setuptools import setup

setup(name='fae',
      version='0.1',
      description='Facial animation evaluation suite',
      author='Dino Vougioukas',
      author_email='dinovgk@gmail.com',
      url='https://github.com/DinoMan/facial-animation-evaluation.git',
      download_url='https://github.com/DinoMan/facial-animation-evaluation/archive/refs/tags/v0.1.tar.gz',
      license='MIT',
      packages=['fae'],
      package_dir={'fae': 'fae'},
      package_data={'fae': ['resources/*']},
      install_requires=[
          'torch',
          'numpy',
          'opencv-python',
          'jiwer',
          'gdown',
          'dtk'
      ],
      entry_points={'console_scripts': ['get_fae_metrics = fae.evaluate:evaluate']},
      zip_safe=False)
