import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="ChanelViT",
    py_modules=["channelvit"],
    version="1.0",
    description="Channel Vision Transformers",
    author="Insitro",
    packages=find_packages() + ["channelvit/config"],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
