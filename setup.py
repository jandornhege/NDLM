from setuptools import setup, find_packages

setup(
    name="ndlm",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["rdflib>=7.0"],
)