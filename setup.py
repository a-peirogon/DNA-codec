from setuptools import setup, find_packages

setup(
    name="dna_codec",
    version="1.0.0",
    description="DNA-based data storage codec with Reed-Solomon error correction",
    packages=find_packages(include=["dna_codec", "dna_codec.*"]),
    install_requires=["reedsolo>=1.7"],
    python_requires=">=3.9",
)
