try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='kito',
    version='1.0.0',
    author='Roman Sol (ZFTurbo)',
    packages=['kito', ],
    url='https://github.com/ZFTurbo/Keras-inference-time-optimizer',
    license='MIT License',
    description='Keras inference time optimizer',
    long_description='',
    install_requires=[
        'keras',
        "numpy",
    ],
)
