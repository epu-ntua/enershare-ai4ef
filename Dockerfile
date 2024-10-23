# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /leif_app/
ENV DAGSTER_HOME=/leif_app

# Copy the current directory contents into the container at /leif_app
COPY ./ai4ef_train_app /leif_app/ai4ef_train_app
COPY ./ai4ef_model_app /leif_app/ai4ef_model_app/
COPY ./shared_storage/ /leif_app/shared_storage/


# Install build-essential and other necessary system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
# Install any needed packages specified in requirements.txt and cleanup temp files
# RUN apt update && apt upgrade --no-cache-dir
RUN pip install -r /leif_app/shared_storage/python_requirements.txt \ 
    && rm -rf /var/cache/apk/* /tmp/*