from setuptools import setup

setup(name='ulmfit_bpe',
      version='1.0',
      description='ULMFiT version of TrollBlock project model',
      author='TrollBlock',
      license='MIT',
      packages=[
            'ulmfit_bpe',
            'ulmfit_bpe/model',
      ],
      install_requires=[
            'pandas==0.24.2',
            'tqdm==4.23.4',
            'torch==1.1.0',
            'scikit-learn==0.20.3',
            'fastai==1.0.52',
            'numpy==1.22.0',
            'sentencepiece==0.1.82'
      ],
      zip_safe=False)
