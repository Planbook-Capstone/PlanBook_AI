from setuptools import setup, find_packages

setup(
    name="fastapi-cli",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "fastapi=cli.main:main",
        ],
    },
    install_requires=[
        "fastapi",
        "uvicorn",
    ],
)
