from setuptools import setup, find_packages

setup(
    name='drum_sound_classifier',
    version='1.0.0',
    packages=find_packages(include=['drum_sound_classifier']),
    install_requires=[
        'audioread',
        'h5py',
        'librosa',
        'numpy',
        'pandas',
        'pytorch-ignite',
        'scipy',
        'scikit-learn',
        'tables',
        'tensorboard',
        'tensorboardX',
        'torch',
        'tqdm'
    ],
    python_requires='>=3.6',
    author='Evan Radkoff',
    description="Train and compare different models for classifying drum sounds",
)
