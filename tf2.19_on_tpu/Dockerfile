# Use a specific Python version for reproducibility
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install TensorFlow for TPU and other dependencies
# Pinning versions is a good practice for stable builds.
RUN pip install --no-cache-dir \
    tensorflow==2.18.0 \
    tensorflow-datasets==4.9.4

# Copy the training script into the container
COPY main.py .

# Command to run the training script when the container starts
CMD ["python", "main.py"]
