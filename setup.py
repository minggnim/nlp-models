from setuptools import find_packages, setup

setup(
    name='bert-classifier',
    packages=find_packages(where='src.bert_classifier'),
    package_dir={'': 'src'},
    version='0.1.0',
    description='Bert based classification model',
    author='ming_gao@outlook.com',
    license='MIT',
    install_requires=[
        'torch',
        'transformers',
    ]
)