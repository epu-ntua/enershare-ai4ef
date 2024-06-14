from setuptools import find_packages, setup

with open("python_requirements.txt", "r") as f:
    python_requirements = f.read().splitlines()

setup(
    name="dagster_ai4ef_train_app",
    packages=find_packages(exclude=["dagster_ai4ef_train_app_tests"]),
    install_requires=python_requirements,
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
