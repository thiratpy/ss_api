# Use an official, stable PyTorch base image
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-runtime

# Install pip, git, and a more comprehensive set of common system libraries for vision packages
# Using --no-install-recommends to keep the image size down
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install dependencies using pip, ensuring numpy is handled correctly with PyTorch's version
# Upgrading pip and setting a longer timeout for robustness
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt --timeout=600

# Copy the rest of your application code
COPY . .

# Command to run your worker when the container starts
CMD ["python", "-u", "runpod_worker.py"]