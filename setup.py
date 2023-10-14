from setuptools import setup, find_packages

setup(
    name="naive-flow",
    version="0.3.1",
    packages=find_packages(),
    license="MIT",
    description="Naive Flow, lightweight and obstructive higher level framework based on Pytorch",
    install_requires=[
        "pydantic >= 2.0",
        "pydantic-settings >= 2.0",
    ],
)
