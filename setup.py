from setuptools import setup, find_packages

setup(
    name='smcmodel_localize',
    packages=find_packages(),
    version='0.0.1',
    include_package_data=True,
    description='Define a sequential Monte Carlo (SMC) model to estimate positions of objects from sensor data',
    long_description=open('README.md').read(),
    url='https://github.com/WildflowerSchools/smcmodel_localize',
    author='Ted Quinn',
    author_email='ted.quinn@wildflowerschools.org',
    install_requires=[
        'tensorflow==1.13.1',
        'tensorflow-probability==0.6.0',
        'numpy==1.16.2'
    ],
    keywords=['bayes', 'smc'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
    ]
)
