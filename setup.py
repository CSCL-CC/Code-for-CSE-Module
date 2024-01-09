from setuptools import setup, find_packages


if __name__ == '__main__':
    setup(
        name='CSCL_CSE',
        verson='1.0.2',
        author='ACM MM Submission 106 Anonymous Authors',
        description='Training the Key CSE Module of the CSCL Framework',
        packages=find_packages(),
        install_requires=[
            'numpy>=1.19.2',
            'pandas>=1.1.3',
        ]
    )
