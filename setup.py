from setuptools import setup, find_packages

setup(
    name="navier_stokes",
    version="0.1.0",  
    description="for solving and comparing different solutions to 1D Navier Stokes", 
    author="Hengrui Zhu, Doyup Kwon, Marie Joe Sawma, Zhan Wu and Maria Fleury",
    package_dir={"": "src"},  
    packages=find_packages(where="src"),  
    python_requires=">=3.9, <4", # tuple typing was introduced in Python 3.9
    install_requires=["numpy", "matplotlib"],
)