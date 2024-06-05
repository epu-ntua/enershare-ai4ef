# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /leif_app/

# Copy the current directory contents into the container at /leif_app
COPY ./python_requirements.txt /leif_app/
COPY ./api /leif_app/api/
COPY ./datasets /leif_app/datasets/
COPY ./json_files /leif_app/json_files/
COPY ./models-scalers/ /leif_app/models-scalers/
COPY ./postgrest-openapi-ro.yaml/ /leif_app/

# Install build-essential and other necessary system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*
    
# Install any needed packages specified in requirements.txt and cleanup temp files
# RUN apt update && apt upgrade
RUN pip install --no-cache-dir -r /leif_app/python_requirements.txt \
    && rm -rf /var/cache/apk/* /tmp/*