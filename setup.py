"""
ClaudeDex Trading Bot - Setup Configuration
Advanced DexScreener trading bot with ML ensemble and MEV protection
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
def read_requirements(file):
    """Read requirements from file"""
    if os.path.exists(file):
        with open(file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="claudedex-trading-bot",
    version="1.0.0",
    author="ClaudeDex Team",
    author_email="team@claudedex.io",
    description="Advanced multi-chain DEX trading bot with ML ensemble and MEV protection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/claudedex/trading-bot",
    project_urls={
        "Bug Tracker": "https://github.com/claudedex/trading-bot/issues",
        "Documentation": "https://docs.claudedex.io",
        "Source": "https://github.com/claudedex/trading-bot",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Framework :: AsyncIO",
    ],
    packages=find_packages(exclude=["tests*", "docs*", "scripts*", "kubernetes*"]),
    python_requires=">=3.11",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("test-requirements.txt"),
        "ml": [
            "tensorflow>=2.13.0",
            "torch>=2.0.0",
            "transformers>=4.30.0",
        ],
        "monitoring": [
            "prometheus-client>=0.17.0",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "claudedex=main:main",
            "claudedex-setup=scripts.init_config:main",
            "claudedex-db-init=scripts.setup_database:main",
            "claudedex-health=scripts.health_check:main",
        ],
    },
    include_package_data=True,
    package_data={
        "config": ["*.yaml", "*.json"],
        "kubernetes": ["*.yaml"],
        "docs": ["*.md"],
    },
    zip_safe=False,
    keywords=[
        "trading", "cryptocurrency", "dex", "defi", "bot",
        "machine-learning", "ethereum", "bsc", "polygon",
        "mev", "arbitrage", "dexscreener", "web3"
    ],
)