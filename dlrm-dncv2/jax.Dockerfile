# Use the official Python 3.12 image, as required.
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install git, which is required for cloning the recml repository
RUN apt-get update && apt-get install -y git --no-install-recommends && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel and a modern version of setuptools first.
# This ensures the build environment is correct for Python 3.12 and avoids distutils errors.
RUN pip install --no-cache-dir --upgrade pip wheel setuptools

# Copy the new requirements file into the container
COPY requirements.txt .

# Install all dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U "jax[tpu]>0.4.23" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Clone the recml repo and install it from source to ensure the latest version is used.
# TODO: Please replace with the correct internal git URL for your project if needed.
RUN git clone https://github.com/AI-Hypercomputer/RecML.git /app/recml
RUN pip install -e /app/recml

# Copy your application code into the container
COPY dlrm_experiment.py .
COPY main.py .

# Define the command to run when the container starts
CMD ["python", "main.py"]

