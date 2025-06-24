# Use a standard Python 3.10 image as the base
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training script into the container
COPY train.py .

# Set the command to run when the container starts
CMD ["python", "train.py"]
