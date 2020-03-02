import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sle",
    version="0.0.1",
    author="Leon Reteig",
    author_email="leonreteig@gmail.com",
    description="Multivariate analysis code for SLE project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lcreteig/SLE",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    python_requires='>=3.7',
)
