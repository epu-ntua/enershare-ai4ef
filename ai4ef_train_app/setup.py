from setuptools import find_packages, setup
import os

# Get the current working directory
current_directory = os.getcwd()

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Construct the path to the requirements.txt file in the parent directory
requirements_path = os.path.join(parent_directory, 'python_requirements.txt')

with open(requirements_path, "r") as f:
    python_requirements = f.read().splitlines()

setup(
    name="ai4ef_train_app",
    packages=find_packages(exclude=["ai4ef_train_app_tests"]),
    install_requires=python_requirements,
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
