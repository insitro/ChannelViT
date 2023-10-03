import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="AMLSSL",
    py_modules=["amlssl"],
    version="1.0",
    description="AML toolkit for self supervised learning",
    author="Insitro",
    packages=find_packages() + ["amlssl/config"],
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    # extras_require={'dev': ['pytest']},
)
