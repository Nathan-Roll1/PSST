import os
import pkg_resources
from setuptools import setup, find_packages


setup(
    name="psst",
    py_modules=["psst"],
    version=read_version(),
    description="Prosodic Speech Segmentation with Transformers",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    readme="README.md",
    python_requires=">=3.7",
    author="Nathan Roll",
    url="https://github.com/Nathan-Roll1/PSST",
    license="MIT",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)