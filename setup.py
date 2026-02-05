#!/usr/bin/env python3
"""
BBAC ICS Framework - Python Package Setup
Python 3.10 ONLY (ROS2 Humble compatibility)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = 'BBAC ICS Framework - Behavioral-Based Access Control for Industrial Control Systems'

# Core requirements with strict versions for ROS2 Humble / Python 3.10
CORE_REQUIREMENTS = [
    # Core ML (ROS2 Humble compatible versions)
    'numpy>=1.23.0,<1.25.0',      # Compatible with scipy + sklearn + tensorflow
    'scipy>=1.10.0,<1.12.0',      # Explicit version (avoid apt conflicts)
    'pandas~=2.1',                # Stable data manipulation
    'pyyaml>=6.0',
    
    # Machine Learning
    'scikit-learn~=1.3.0',        # Allow patches (1.3.1, 1.3.2) but not 1.4.x
]

# Visualization requirements
VISUALIZATION_REQUIREMENTS = [
    'matplotlib~=3.8',
    'seaborn~=0.13',
    'plotly~=5.18',
]

# Development requirements
DEV_REQUIREMENTS = [
    'pytest>=7.4.0',
    'black>=23.0.0',
    'flake8>=6.0.0',
    'mypy>=1.4.0',
]

# Optional ML requirements (future LSTM models)
ML_REQUIREMENTS = [
    'tensorflow>=2.13.0,<2.16.0',  # TF 2.15.x for Python 3.10
    'keras>=2.13.0,<2.16.0',
]

# Combine all requirements
ALL_REQUIREMENTS = (
    CORE_REQUIREMENTS + 
    VISUALIZATION_REQUIREMENTS
)

setup(
    name='bbac-framework',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Behavioral-Based Access Control Framework for ICS',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/bbac-framework',
    
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: System :: Systems Administration :: Authentication/Directory',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    
    python_requires='==3.10.*',  # Strict: Python 3.10 only (ROS2 Humble)
    
    install_requires=ALL_REQUIREMENTS,
    
    extras_require={
        'dev': DEV_REQUIREMENTS,
        'ml': ML_REQUIREMENTS,
        'all': ALL_REQUIREMENTS + DEV_REQUIREMENTS + ML_REQUIREMENTS,
    },
    
    entry_points={
        'console_scripts': [
            # Experimentos
            'bbac-ablation=experiments.ablation_study:main',
            'bbac-baseline=experiments.baseline_comparison:main',
            'bbac-adaptive=experiments.adaptive_eval:main',
            
            # Training
            'bbac-train=scripts.train_models:main',
            
            # Runner
            'bbac-run=util.runner:main',
        ],
    },
    
    include_package_data=True,
    package_data={
        'bbac_framework': [
            'config/*.yaml',
        ],
    },
    
    zip_safe=False,
)