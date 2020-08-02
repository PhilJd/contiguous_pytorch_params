from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
   name="contiguous_params",
   version="1.0",
   description="Make pytorch parameters contiguous to speed up training by 100x.",
   license="Apache 2.0",
   long_description=long_description,
   author="Philipp Jund",
   author_email="ijund.phil@gmail.com",
   url="http://www.github.com/philjd/contiguous_pytorch_params",
   packages=["contiguous_params"],
   keywords="pytorch contiguous parameters speed up accelerate",
)