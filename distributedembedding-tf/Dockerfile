# Use an official TensorFlow base image
FROM tensorflow/tensorflow:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script into the container
COPY main.py .

# Command to run when the container starts
CMD ["python", "main.py"]
