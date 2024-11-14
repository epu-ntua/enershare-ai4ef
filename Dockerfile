# Dockerfile

# ---- Base Stage ----
FROM python:3.10.12-slim AS base

# Set the working directory in the container
WORKDIR /leif_app/
ENV DAGSTER_HOME=/leif_app

# Install build-essential and other necessary system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file to leverage Docker cache for dependencies
COPY ./shared_storage/python_requirements.txt /leif_app/shared_storage/python_requirements.txt

# Install Python dependencies
RUN pip install -r /leif_app/shared_storage/python_requirements.txt \
    && rm -rf /var/cache/apk/* /tmp/*

# ---- Final Stage ----
FROM base AS final

# Set the working directory in the final stage
WORKDIR /leif_app/

# Copy application code from the current directory into the container
COPY ./ai4ef_train_app /leif_app/ai4ef_train_app
COPY ./ai4ef_model_app /leif_app/ai4ef_model_app/
COPY ./shared_storage/ /leif_app/shared_storage/