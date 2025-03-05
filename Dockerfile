# Base image with CUDA support
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

# Set the Python version and install basic dependencies

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv

# Set the working directory in the container
WORKDIR /app

COPY requirements.txt .


# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory's contents into the container
COPY . /app

# Default command (can be overridden)
CMD ["fastapi", "run", "api.py"]