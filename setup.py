import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modev-pablorosado", # Replace with your own username
    version="0.0.1",
    author="Pablo Rosado",
    author_email="pabloarosado@gmail.com",
    description="Model Development for Data Science Projects.",
    keywords="data science model development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pabloarosado/modev",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'plotly',
        'scikit-learn',
        'tqdm',
    ],
    include_package_data=True,
    package_data={'modev': ['data/*.csv']},
    python_requires='>=3.6',
)
