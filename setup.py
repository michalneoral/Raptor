from setuptools import setup, find_packages

print(find_packages())

setup(
    name='raptor',
    version='1.0',
    description='Raptor code by Michal Neoral',
    packages=find_packages(),
    install_requires=[
        'tqdm',
        'opencv-python',
        'imageio==2.13.5',
        'wandb',
        'pypng',
        'gdown',
        'kornia',
        'timm',
#        'numpy>=1.21',
#        'nptyping',
    ]#,
)
