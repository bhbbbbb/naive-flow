from setuptools import setup, find_packages

setup(
    name="pytorch-model-utils",
    version="0.2.0",
    packages=find_packages(),
    license="MIT",
    description="Utils for pytorch",
    install_requires=[
        "pydantic >= 2.0",
        "pydantic-settings >= 2.0",
    ],
)
