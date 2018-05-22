from setuptools import setup, find_packages

setup(
    name='Feature Finder',
    version='1.0.0',
    author='Catherine Noll',
    author_email='noll.catherine@gmail.com',
    description='Find predictive features',
    packages=find_packages(),
    install_requires=['nose==1.3.7',
                      'pandas==0.23.0',
                      'scikit-learn==0.19.1',
                      'seaborn==0.8.1',
                      'matplotlib==2.2.2'],
    entry_points={
        'console_scripts': [
            'feature-finder = feature_finder.__main__:main'
        ]
    },
)
