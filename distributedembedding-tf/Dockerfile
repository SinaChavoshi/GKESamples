# Use a more specific, smaller base image if possible
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# 1. Copy only the requirements file first.
# This layer only changes if you edit requirements.txt
COPY requirements.txt .

# 2. Install dependencies.
# This slow step will now be cached and re-used as long as
# requirements.txt doesn't change.
RUN pip install --no-cache-dir -r requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# 3. Copy the rest of your application code.
# This is the part that changes most often. Now, only this layer
# and subsequent ones will be rebuilt when you edit your code.
COPY train_jax.py .

# Set the command to run when the container starts
CMD ["python", "train_jax.py"]
