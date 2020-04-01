try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='kito',
    version='1.0.4',
    author='Roman Sol (ZFTurbo)',
    packages=['kito', ],
    url='https://github.com/ZFTurbo/Keras-inference-time-optimizer',
    license='MIT License',
    description='Keras inference time optimizer',
    long_description='This code takes on input trained Keras model and optimize layer structure and weights '
                     'in such a way that model became much faster (~10-30%), but works identically to '
                     'initial model. It can be extremely useful in case you need to process large amount '
                     'of images with trained model. Reduce operation was tested on all Keras models zoo. '
                     'More details: https://github.com/ZFTurbo/Keras-inference-time-optimizer',
    install_requires=[
        'keras',
        "numpy",
    ],
)
