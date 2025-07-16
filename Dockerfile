# Use an official RunPod base image with PyTorch and CUDA pre-installed
FROM runpod/pytorch:2.2.1-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir makes the image smaller
# The timeout is increased in case mediapipe or torch takes a while
RUN pip install --no-cache-dir -r requirements.txt --timeout=300

# Copy the rest of your application code (the worker script and model directory)
COPY . .

# Command to run your worker when the container starts
CMD ["python", "-u", "runpod_worker.py"]
# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir makes the image smaller
# The timeout is increased in case mediapipe or torch takes a while
RUN pip install --no-cache-dir -r requirements.txt --timeout=300

# Copy the rest of your application code (the worker script and model directory)
COPY . .

# Command to run your worker when the container starts
CMD ["python", "-u", "runpod_worker.py"]